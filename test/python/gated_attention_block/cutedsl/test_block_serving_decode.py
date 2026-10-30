# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's DECODE (serving) form: a dense ``[B, S_q <= 32]`` declaration with ``kv_source="paged"`` over a
paged KV cache -- the step's tokens written through, then stage (4) over the first ``kv_lens[b]`` keys of every sequence's
``block_table`` row on the Rubin d256 decode tile in ONE token unit where its width table holds the packed rows (PackGQA over
the whole group, the decode split model's split-KV with fp32 partials in the workspace, the sigmoid gate in the split COMBINE
under ``fuse_gate``), past the table on the paged d256 PREFILL tile with the group packed (the KV streamed once; the gate in
that body's epilogue) wherever that tile packs the group, else the decode tile's token-unit and unpacked arms (the route per
policy is ``test_block_serving_mtp_route.py``'s business), the decode tile UNPACKED past the packed routes up to ``S_q = 32``, and a
typed decline past 32 tokens.

Host cells (any device, the declaration cells even on the CPU): the constants and the route table per ``(group, S_q)`` at the
four served groups (16 = 32 / 2, 12 = 24 / 2, 8 = 8 / 1, 6 = 24 / 4) against the module's route table; the static
``serving_envelope`` -- every decline the declaration raises is predicted by it, on this part and on a Rubin reading; the
declaration rows (typed, naming the feature); the declaration FACTS (route, split, where the gate runs, the stage list, the
capture-safe projections); the execute-time refusals on an UNCOMPILED block -- the Q-side length tensor ``seq_lens`` REFUSED
under ``kv_source="paged"``, ``block_table`` + ``kv_lens`` REQUIRED, form checks -- under ``set_sync_debug_mode("error")``;
the workspace carve unchanged by the read.

cc 10.0 cells (``requires_blackwell``, 2026-10-29; the SM100 line of ``DENSE_SERVING_CCS``, functional only): the same oracle matrix at
every served group x ``S_q`` 1 .. 5 with ``fuse_gate=False`` (the sm100 row claims no gate epilogue: stage (5) its own launch) on the
SM100 decode tile (``sm100/decode_d256_f16.py``: the 16-column tile packed, the 16-column tile UNPACKED at groups 6 / 12 past 16 rows,
the paged prefill tile packed at groups 16 / 8 / 4 -- the line's own rule, pinned on the compiled module), the 27B geometry (24 / 4 at
``d_model`` 5120) at ``S_q`` 1 / 4, the pool matrix, capture / replay, two launches; the Rubin-only knobs declined by name there.

Rubin cells (one compiled block per cell, cached): every served group x ``S_q`` 1 .. 5 against the per-sequence fp32 oracle
(the block run on the FULL sequence, its last ``S_q`` rows; the prefix K / V of the pools ARE the oracle's bf16 rows), with
``kv_lens > S_q`` on every live row, a tail tile, one KV tile, ring depth 3 and 4, a padded entry (``kv_lens = 0`` + block 0 +
slot -1 -> ``out`` rows exactly 0, block 0 untouched -- the serving stacks' padded full-graph entry), ``S_q = 5`` on the
packed PREFILL tile at groups 16 / 8, the UNPACKED decode tile at 12 and the 32-column tile at 6 / 4, the route pinned against the
compiled module; paged == dense on the same tokens (the standalone adapter over a padded dense cache gathered from the pools: BITWISE);
the block's form vs the standalone adapter on the SAME Q views / pools / table / gate: BITWISE; split == unsplit within the
combine's derived two-term budget (the pre-gate O of the block's own stage; LSE to fp32 rounding); the gate in the combine vs
the separate gate launch within the stated one-rounding budget (the exact max diff printed); L1 / L2 / L3 pools holding the
same values BITWISE on every page size (16 / 32 / 64 / 128) and f16; a block table that is a row SLICE of a larger table at an
arbitrary row stride (bitwise the compact table); CUDA-graph capture of the FIRST execute on a NEW stream with no warm-up and
the replay with new lengths / tables / slots / inputs bitwise the eager run; ``get_workspace_size()`` honest (the split
partials inside, a sentinel past the carve untouched); two launches bitwise.

Both-parts cells (``requires_dense_part``, 2026-10-29; the end of the module): a serving stack's PACKED K|V cache -- ONE ``[pages, 4, P, 512]``
tensor, K in columns ``[0, D)`` and V in ``[D, 2D)`` of every row -- bound as the two half views at the 24 / 4 geometry, page 16 with one
token per sequence and page 32 with a four-token step, kv to 2048 with a padded and a short entry, int64 slots, bf16 and f16: the oracle,
the write-through in the right halves with every other byte untouched, two launches bitwise, bitwise the HND-compact twin.
"""

import contextlib
import math
import os
import sys
import time

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    DECODE_CLAIMED_MAX_N_Q,
    DECODE_CLAIMED_MAX_S_Q,
    DECODE_MAX_ROWS,
    DECODE_MAX_S_Q,
    DECODE_PREFILL_ROUTE_MIN_CTAS,
    DECODE_ROUTE_POLICIES,
    DECODE_ROUTE_SINGLE_UNIT,
    DECODE_ROUTE_TOKEN_UNITS,
    DECODE_WIDE_TILE_CCS,
    DENSE_SERVING_CCS,
    FP8_PROJ_BF16_ATTENTION_CCS,
    KV_SOURCE_PAGED,
    KV_SOURCE_SELF,
    QUANT_MODE_FP8_PROJ_BF16_ATTENTION,
    SERVING_PAGE_SIZES,
    SERVING_POOL_LAYOUTS,
    DecodeEnvelope,
    DecodeRoute,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    QuantSpec,
    ServingEnvelope,
    decode_route,
)
from cudnn.gated_attention_block import api as _api  # noqa: E402  (the device read the cc 10.0 host rows patch)
from cudnn.gated_attention_block.api import (
    DECODE_TILE,
    PREFILL_TILE,
    _CacheWrite,
    _cols,
    _Projection,
    _Quantize,
    _Sdpa,
    _SigmoidGate,
    _view,
    decode_split_choice,
)

requirement_error = cutedsl_requirement_error("the gated attention block's serving tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107
requires_blackwell = pytest.mark.requires_blackwell  # the registered marker: an admitted Blackwell-line part of DENSE_SERVING_CCS (cc 10.0)
requires_dense_part = pytest.mark.requires_dense_part  # the registered marker: EVERY served part of DENSE_SERVING_CCS (cc 10.7 and cc 10.0)
_SM100 = (10, 0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    amax_scale,
    build_rope_tables,
    gated_attention_block_fp8_reference,
    gated_attention_block_reference,
    quant_e4m3,
)

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SM107 = (10, 7)
_D = 256
_ROPE = 64
# The five served groups (h_q / h_kv): the 397B TP 1 shard (32 / 2), Flash-Next TP 1 (24 / 2), the 397B TP 4 shard (8 / 1), the 27B
# members (24 / 4), the 397B TP 8 shard (4 / 1) -- at a shrunk d_model so the projections stay cheap; every width a multiple of the
# forced 256-wide GEMM tile.  The 397B TP 2 shard (16 / 1) is group 16 with ONE KV head: its own cells sit in _SHARDS below.
_GROUPS = {16: (32, 2), 12: (24, 2), 8: (8, 1), 6: (24, 4), 4: (4, 1)}
_D_MODEL = 512
_LAYOUTS = SERVING_POOL_LAYOUTS
# The route table per POLICY, (group, S_q) -> (tile, N_Q, token units, packed), read WITHOUT a batch (= below the fill).  "single_unit"
# (the default): the decode tile's single-unit widths (16, 32, 64) hold the packed rows in ONE unit (the 64-column width for every
# packed group of the DECODE form: paged under the bottom-right causal mask); past them the paged d256 PREFILL tile with the group
# packed wherever that tile packs the group (a group dividing its 128 rows: 16 at S_q 5 .. 8) -- but ONLY where the step FILLS the
# part (``batch x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS``: the prefill tile is one unsplit CTA per sequence and KV head) or no two-unit
# decode route holds it; below the fill, and for a group the prefill tile cannot pack (12 = 24 / 2 at S_q 6 .. 8), the decode tile's
# token-unit arm (up to two units of the 64-column tile), and its UNPACKED arm (one head per CTA, ``group`` streams) past that -- the
# arm every group falls back to up to 32 tokens (32 / 2 from S_q 9: 144 packed rows, past the prefill tile and a third unit).
# "token_units": the adapter's own route verbatim (the widest width in up to two units, then the unpacked arm), whatever the batch.
# Past S_q 32 NOTHING under either policy: the DECODE form's served envelope ends there (``DECODE_MAX_ROWS``), a longer dense step is
# the EXTEND form's packed business.  Every group carries S_q 1 .. 5 (the oracle matrix indexes the table there), 8, 9 and 33.
# ``_FILLED_ROUTES`` overrides the default policy's entries AT the fill (``_fill_batch(group)`` sequences): the prefill tile.
_P, _T = PREFILL_TILE, DECODE_TILE
_ROUTE_TABLES = {
    DECODE_ROUTE_SINGLE_UNIT: {
        16: {
            1: (_T, 16, 1, True),
            2: (_T, 32, 1, True),
            3: (_T, 64, 1, True),
            4: (_T, 64, 1, True),
            5: (_T, 64, 2, True),
            8: (_T, 64, 2, True),
            9: (_T, 16, 1, False),
            33: (None, 0, 0, False),
        },
        12: {
            1: (_T, 16, 1, True),
            2: (_T, 32, 1, True),
            3: (_T, 64, 1, True),
            4: (_T, 64, 1, True),
            5: (_T, 64, 1, True),
            6: (_T, 64, 2, True),
            8: (_T, 64, 2, True),
            9: (_T, 64, 2, True),
            11: (_T, 16, 1, False),
            33: (None, 0, 0, False),
        },
        8: {
            1: (_T, 16, 1, True),
            2: (_T, 16, 1, True),
            3: (_T, 32, 1, True),
            4: (_T, 32, 1, True),
            5: (_T, 64, 1, True),
            8: (_T, 64, 1, True),
            9: (_T, 64, 2, True),
            17: (_T, 32, 1, False),
            33: (None, 0, 0, False),
        },
        6: {
            1: (_T, 16, 1, True),
            2: (_T, 16, 1, True),
            3: (_T, 32, 1, True),
            4: (_T, 32, 1, True),
            5: (_T, 32, 1, True),
            6: (_T, 64, 1, True),
            8: (_T, 64, 1, True),
            9: (_T, 64, 1, True),
            21: (_T, 32, 1, False),
            33: (None, 0, 0, False),
        },
        4: {
            1: (_T, 16, 1, True),
            2: (_T, 16, 1, True),
            3: (_T, 16, 1, True),
            4: (_T, 16, 1, True),
            5: (_T, 32, 1, True),
            8: (_T, 32, 1, True),
            9: (_T, 64, 1, True),
            17: (_T, 64, 2, True),
            33: (None, 0, 0, False),
        },
    },
    DECODE_ROUTE_TOKEN_UNITS: {
        16: {
            1: (_T, 16, 1, True),
            2: (_T, 32, 1, True),
            3: (_T, 64, 1, True),
            4: (_T, 64, 1, True),
            5: (_T, 64, 2, True),
            8: (_T, 64, 2, True),
            9: (_T, 16, 1, False),
            33: (None, 0, 0, False),
        },
        12: {
            1: (_T, 16, 1, True),
            2: (_T, 32, 1, True),
            3: (_T, 64, 1, True),
            4: (_T, 64, 1, True),
            5: (_T, 64, 1, True),
            6: (_T, 64, 2, True),
            8: (_T, 64, 2, True),
            9: (_T, 64, 2, True),
            11: (_T, 16, 1, False),
            33: (None, 0, 0, False),
        },
        8: {
            1: (_T, 16, 1, True),
            2: (_T, 16, 1, True),
            3: (_T, 32, 1, True),
            4: (_T, 32, 1, True),
            5: (_T, 64, 1, True),
            8: (_T, 64, 1, True),
            9: (_T, 64, 2, True),
            17: (_T, 32, 1, False),
            33: (None, 0, 0, False),
        },
        6: {
            1: (_T, 16, 1, True),
            2: (_T, 16, 1, True),
            3: (_T, 32, 1, True),
            4: (_T, 32, 1, True),
            5: (_T, 32, 1, True),
            6: (_T, 64, 1, True),
            8: (_T, 64, 1, True),
            9: (_T, 64, 1, True),
            21: (_T, 32, 1, False),
            33: (None, 0, 0, False),
        },
        4: {
            1: (_T, 16, 1, True),
            2: (_T, 16, 1, True),
            3: (_T, 16, 1, True),
            4: (_T, 16, 1, True),
            5: (_T, 32, 1, True),
            8: (_T, 32, 1, True),
            9: (_T, 64, 1, True),
            17: (_T, 64, 2, True),
            33: (None, 0, 0, False),
        },
    },
}
# The default policy AT the fill (``decode_route(batch=_fill_batch(group))``): the paged prefill tile wherever the group divides its 128
# rows and the step is past the single-unit table; every other cell is the table above.
_FILLED_ROUTES = {
    16: {5: (_P, 0, 0, True), 8: (_P, 0, 0, True)},
    8: {9: (_P, 0, 0, True)},
    4: {17: (_P, 0, 0, True)},
}


def _fill_batch(group):
    """The smallest batch at which a step of ``group`` fills the part for the prefill route: ``batch x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS``."""
    return -(-DECODE_PREFILL_ROUTE_MIN_CTAS // _GROUPS[group][1])


# The route before the 64-column width landed (the measured widths (16, 32)): what ``decode_q_tile_max=32`` restores, value for value --
# the perf A/B's "develop behaviour" arm.  Group 16 / 8 at S_q 3 .. 4 / 5 .. 8 rode two units of the 32-column tile under "token_units"
# and the prefill tile under "single_unit" where the group divides 128; group 12 past S_q 4 the UNPACKED arm (12 streams).  Under the
# default policy this is the reading AT the fill; below it the capped route takes two units of the 32-column tile wherever two hold the
# step (``_ROUTE_TABLE_CAP_32_BELOW_FILL``: 16 at S_q 3 / 4, 8 at 5 .. 8, 4 at 9 .. 16) and the prefill tile where they do not.
_ROUTE_TABLE_CAP_32 = {
    DECODE_ROUTE_SINGLE_UNIT: {
        16: {1: (_T, 16, 1, True), 2: (_T, 32, 1, True), 3: (_P, 0, 0, True), 4: (_P, 0, 0, True), 5: (_P, 0, 0, True), 8: (_P, 0, 0, True)},
        12: {
            1: (_T, 16, 1, True),
            2: (_T, 32, 1, True),
            3: (_T, 32, 2, True),
            4: (_T, 32, 2, True),
            5: (_T, 16, 1, False),
            6: (_T, 16, 1, False),
            8: (_T, 16, 1, False),
            9: (_T, 16, 1, False),
        },
        8: {3: (_T, 32, 1, True), 4: (_T, 32, 1, True), 5: (_P, 0, 0, True), 8: (_P, 0, 0, True), 9: (_P, 0, 0, True)},
        6: {5: (_T, 32, 1, True), 6: (_T, 32, 2, True), 8: (_T, 32, 2, True), 9: (_T, 32, 2, True)},
        4: {9: (_P, 0, 0, True), 17: (_P, 0, 0, True)},
    },
    DECODE_ROUTE_TOKEN_UNITS: {
        16: {3: (_T, 32, 2, True), 4: (_T, 32, 2, True), 5: (_T, 16, 1, False), 8: (_T, 16, 1, False)},
        12: {3: (_T, 32, 2, True), 4: (_T, 32, 2, True), 5: (_T, 16, 1, False), 6: (_T, 16, 1, False), 8: (_T, 16, 1, False), 9: (_T, 16, 1, False)},
        8: {5: (_T, 32, 2, True), 8: (_T, 32, 2, True), 9: (_T, 16, 1, False)},
        6: {6: (_T, 32, 2, True), 8: (_T, 32, 2, True), 9: (_T, 32, 2, True)},
        4: {9: (_T, 32, 2, True), 17: (_T, 32, 1, False)},
    },
}
_ROUTE_TABLE_CAP_32_BELOW_FILL = {
    16: {3: (_T, 32, 2, True), 4: (_T, 32, 2, True)},
    8: {5: (_T, 32, 2, True), 8: (_T, 32, 2, True)},
    4: {9: (_T, 32, 2, True)},
}
_ROUTE_TABLE = _ROUTE_TABLES[DECODE_ROUTE_SINGLE_UNIT]
# THE SM100 LINE (cc 10.0; 2026-10-29): (tile, n_q, token units, packed, kv_streams) per (group, S_q) at the SMALL-LAUNCH reading (no
# batch / no KV bound: never a wide launch) -- the SDPA adapter's own rule off cc 10.7, input for input (config_sm100.decode_d256_q_tile
# with the wide-tile reading of config_sm100.d256_wide_decode_launch): packed rows <= 16 ride the 16-column tile in ONE unit; the
# 32-column arm the adapter takes on a WIDE launch (>= 8 units streaming >= 4096 KV tiles: ``_SM100_WIDE`` below) is a typed decline
# naming the lift until its cells run (``DECODE_WIDE_TILE_CCS``, EMPTY; the review follow-up of 2026-10-29); else the paged prefill tile
# packed wherever the group divides 128 (16 / 8 / 4), at EVERY batch (no two-unit arm on the line) under BOTH policies; else the
# 16-column tile UNPACKED (``group`` streams) to 16 tokens; 17 .. 32 tokens unpacked need the 32-column width = a typed decline at every
# launch (``--``: the rule off the wide launch, the same arm on it); 33 the EXTEND form's business as on cc 10.7.  No cap
# (``decode_q_tile_max`` names the Rubin table), ``claimed`` False everywhere (the claimed envelope is the Rubin measurement),
# ``pdl_capable`` on the decode tile.
_U = lambda g: (_T, 16, 1, False, g)  # noqa: E731  -- the 16-column tile unpacked, one head per CTA
_SM100_ROUTE_TABLE = {
    16: {1: (_T, 16, 1, True, 1), **{s: (_P, 0, 0, True, 1) for s in range(2, 9)}, **{s: _U(16) for s in range(9, 17)}, **{s: None for s in range(17, 34)}},
    12: {1: (_T, 16, 1, True, 1), **{s: _U(12) for s in range(2, 17)}, **{s: None for s in range(17, 34)}},
    8: {1: (_T, 16, 1, True, 1), 2: (_T, 16, 1, True, 1), **{s: (_P, 0, 0, True, 1) for s in range(3, 17)}, **{s: None for s in range(17, 34)}},
    6: {1: (_T, 16, 1, True, 1), 2: (_T, 16, 1, True, 1), **{s: _U(6) for s in range(3, 17)}, **{s: None for s in range(17, 34)}},
    4: {**{s: (_T, 16, 1, True, 1) for s in range(1, 5)}, **{s: (_P, 0, 0, True, 1) for s in range(5, 33)}, 33: None},
}
# The wide cells of the line at group 6 (24 / 4, the 27B): (batch, max_kv_len, S_q) -> (the route TODAY, the route once the 32-column arm is
# admitted = ``DECODE_WIDE_TILE_CCS`` gaining (10, 0)): the SDPA adapter's rule takes the 32-column tile packed (W) or unpacked (U32) there and
# the block declines that arm by name until its cells run (None today, the lift in the reason); the small launches of the same steps keep the
# table above.  Units = batch x H_kv packed (4 KV heads), batch x H_q unpacked (24).
_W32, _U32 = (_T, 32, 1, True, 1), (_T, 32, 1, False, 6)
_SM100_WIDE = {
    (2, 65536, 3): (None, _W32),  # 8 packed units x 512 tiles = 4096 tiles: W -- the arm, declined today
    (32, 4096, 3): (None, _W32),  # 128 units x 32 tiles: W -- declined today
    (1, 22528, 17): (None, _U32),  # 24 unpacked units x 176 tiles = 4224: U32 (6 streams) -- declined today
    (4, 5632, 17): (None, _U32),  # 96 x 44 = 4224: U32 -- declined today
    (1, 1024, 3): (_U(6), _U(6)),  # 4 packed units: not wide -> the table's U16 on both readings
    (4, 1024, 17): (None, None),  # 96 x 8 = 768 tiles: not wide -> the unpacked decline on both readings (the width needs the wide rule AND the arm)
}
_WIDE_ARM_LIFT = "the 32-column SM100 decode arm: its cc 10.0 accept cells have not run; admitted when they do (DECODE_WIDE_TILE_CCS gains the part)"
_SENTINEL = 1024.0  # the pools' fill: FINITE (a serving stack's cache is zero-initialized; the tail tile reads and masks the rows past kv_len)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def _rubin_reading(monkeypatch):
    """The RUBIN reading of the route on any host for the host rows that pin the Rubin table's facts on a DECLARED block: since 2026-10-29
    a declaration reads its device's line (a cc 10.0 host records the SM100 table), so these rows pin the part the facts belong to."""
    monkeypatch.setattr(_api, "_route_device_cc", lambda device: None)


@contextlib.contextmanager
def _no_host_sync():
    """Every declaration-time and form check stays on the host without a device sync (Rule 3) -- enforced, not read."""
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _geom(group, heads=None, **over):
    """The block geometry (and the oracle's mirror) of ``group``: the shrunk matrix's heads, or an explicit ``(h_q, h_kv)`` of that
    group (the 397B shards at their real ``d_model``, passed through ``over``)."""
    h_q, h_kv = _GROUPS[group] if heads is None else heads
    assert h_q % h_kv == 0 and h_q // h_kv == group, (group, heads)
    kw = dict(d_model=_D_MODEL, h_q=h_q, h_kv=h_kv, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    kw.update(over)
    return GatedAttentionBlockGeometry(**kw), RefGeometry(**kw)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The 16-bit patterns of a bf16 / f16 tensor: a NaN-safe bitwise comparison (the sentinel is NaN)."""
    return x.contiguous().view(torch.int16)


def _pools(n_pages, h, page_size, d, layout, dtype, device=_DEV, fill=None):
    """Two sentinel-filled pools as ``[P, H, page_size, D]`` views -- ``hnd`` (L1, compact), ``nhd`` (L2, by strides), ``unified``
    (L3: ONE ``[P, H, page_size, 2D]`` content row per token, K at ``+0``, V at ``+D``).  The fill is a large FINITE value
    (``_SENTINEL``): the decode tile's tail tile reads the rows past ``kv_len`` (the last page's stale rows, block 0 for the
    pages a short row does not list) and masks them to ``P = 0`` -- exactly 0 for a finite value, NaN for a NaN one (``0 x
    NaN``); a serving stack's cache is zero-initialized, so finite is the contract and a big sentinel is the detector of an
    unmasked read."""
    fill = _SENTINEL if fill is None else fill
    if layout == "hnd":
        return tuple(torch.full((n_pages, h, page_size, d), fill, device=device, dtype=dtype) for _ in range(2))
    if layout == "nhd":
        return tuple(torch.full((n_pages, page_size, h, d), fill, device=device, dtype=dtype).permute(0, 2, 1, 3) for _ in range(2))
    if layout == "hnd_padded":
        # HND with a PADDED page stride (8 spare token rows per page): covering and 16-B aligned, so the contract admits it; not compact
        return tuple(torch.full((n_pages, h, page_size + 8, d), fill, device=device, dtype=dtype)[:, :, :page_size] for _ in range(2))
    assert layout == "unified"
    row = torch.full((n_pages, h, page_size, 2 * d), fill, device=device, dtype=dtype)
    return row[..., :d], row[..., d:]


def _make_block(group, batch, s_q, *, page=16, layout="hnd", n_pages=64, dtype=torch.bfloat16, device=_DEV, table_row_stride=None, **blk_kw):
    """A declared (not compiled) DECODE block on EMPTY tensors: the declaration cells."""
    geom, _ = _geom(group)
    h_q, h_kv = _GROUPS[group]
    mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=device)  # noqa: E731
    kc, vc = _pools(n_pages, h_kv, page, _D, layout, dtype, device)
    if table_row_stride is None:
        bt = torch.zeros(batch, n_pages, dtype=torch.int32, device=device)
    else:
        bt = torch.zeros(batch, table_row_stride, dtype=torch.int32, device=device)[:, :n_pages]
    kw = dict(paged_kv_page_size=page, kv_source="paged", sample_k_cache=kc, sample_v_cache=vc, sample_block_table=bt)
    kw.update(blk_kw)
    blk = GatedAttentionBlockFwd(
        mk(batch, s_q, _D_MODEL),
        mk(geom.n_qkvg, _D_MODEL),
        torch.ones(_D, dtype=dtype, device=device),
        torch.ones(_D, dtype=dtype, device=device),
        mk(batch, s_q, _ROPE),
        mk(batch, s_q, _ROPE),
        mk(_D_MODEL, h_q * _D),
        mk(batch, s_q, _D_MODEL),
        geom,
        **kw,
    )
    return blk, dict(k_cache=kc, v_cache=vc, block_table=bt, geom=geom)


# ---------------------------------------------------------------------------
# The constants, the route table, the envelope
# ---------------------------------------------------------------------------


def test_constants():
    assert (KV_SOURCE_SELF, KV_SOURCE_PAGED) == ("self", "paged")
    assert SERVING_PAGE_SIZES == (16, 32, 64, 128) and all(p % 16 == 0 for p in SERVING_PAGE_SIZES)
    assert SERVING_POOL_LAYOUTS == ("hnd", "nhd", "unified")
    assert (DECODE_CLAIMED_MAX_S_Q, DECODE_MAX_S_Q, DECODE_CLAIMED_MAX_N_Q) == (4, 32, 32)
    assert DECODE_TILE == "decode_d256_f16" and PREFILL_TILE == "prefill_d256_f16" and DECODE_MAX_ROWS == DECODE_MAX_S_Q == 32
    assert DECODE_PREFILL_ROUTE_MIN_CTAS == 64 and _fill_batch(16) == 32 and _fill_batch(8) == _fill_batch(4) == 64


@pytest.mark.parametrize("policy", [DECODE_ROUTE_SINGLE_UNIT, DECODE_ROUTE_TOKEN_UNITS])
@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_the_route_table_per_group_and_s_q_is_the_adapters_rule(group, policy):
    """The decode route at every served group per policy: the 16- / 32- / 64-column tile in one unit for the packed rows they hold;
    past them the packed PREFILL tile where the group divides its 128 rows (the default policy), else -- and under
    ``token_units`` everywhere -- the 64-column tile in ``ceil(S_q / (64 // G))`` token units up to two, then the UNPACKED arm
    (one head per CTA) up to ``S_q = 32``; NOT SERVED past that (the reason names the EXTEND form); the CLAIMED envelope is the packed DECODE
    tile at ``S_q <= 4`` on the measured widths (16 / 32: ``DECODE_CLAIMED_MAX_N_Q``), the other cells -- the 64-column tile among
    them -- routed but not claimed.  MHA stays unpacked.  ``decode_q_tile_max=32`` (the cap) restores the route before the 64-column
    width landed, value for value, and reads back on the record."""
    geom, _ = _geom(group)
    for s_q, (tile, n_q, units, packed) in _ROUTE_TABLES[policy][group].items():
        r = decode_route(geom, s_q, policy=policy)
        assert isinstance(r, DecodeRoute) and r.s_q == s_q and r.group == group and r.policy == policy and r.q_tile_max is None
        assert (r.tile, r.n_q, r.token_units, r.pack_gqa) == (tile, n_q, units, packed), (group, s_q, policy, r)
        assert r.decode_tile is (tile == DECODE_TILE) and r.prefill_tile is (tile == PREFILL_TILE) and r.served is (tile is not None)
        assert (r.reason == "") is r.served and (r.served or "EXTEND form" in r.reason)
        assert r.claimed is (r.decode_tile and packed and s_q <= DECODE_CLAIMED_MAX_S_Q and n_q <= DECODE_CLAIMED_MAX_N_Q)
        assert r.kv_streams == (units if (r.decode_tile and packed) else group if r.decode_tile else 1 if r.prefill_tile else 0), r
        assert r.batch is None, "no batch given: the small-batch reading"
        if r.decode_tile and packed:
            assert units == -(-s_q // (n_q // group))
        elif r.decode_tile:
            assert units == 1 and s_q <= n_q
        fill = _fill_batch(group)
        if policy == DECODE_ROUTE_SINGLE_UNIT:
            assert r == decode_route(geom, s_q), "the default policy"
            # AT the fill the prefill tile takes the cells it packs; one sequence short of it the route is the batch-less one
            f = decode_route(geom, s_q, policy=policy, batch=fill)
            tf, nf, uf, pf = _FILLED_ROUTES.get(group, {}).get(s_q, (tile, n_q, units, packed))
            assert (f.tile, f.n_q, f.token_units, f.pack_gqa, f.batch) == (tf, nf, uf, pf, fill), (group, s_q, f)
            assert f.prefill_tile is (tf == _P) and f.kv_streams == (1 if tf == _P else r.kv_streams) and f.pdl_capable is (tf == _T), f
            below = decode_route(geom, s_q, policy=policy, batch=fill - 1)
            assert (below.tile, below.n_q, below.token_units, below.pack_gqa, below.batch) == (tile, n_q, units, packed, fill - 1), (group, s_q, below)
            assert below == r, "the batch is a readback, not the route's identity"
        else:
            assert (
                decode_route(geom, s_q, policy=policy, batch=fill) == r and decode_route(geom, s_q, policy=policy, batch=1) == r
            ), "token_units never takes the prefill tile"
        c = decode_route(geom, s_q, policy=policy, max_q_tile=32)
        want = _ROUTE_TABLE_CAP_32[policy].get(group, {}).get(s_q, (tile, n_q, units, packed))
        if policy == DECODE_ROUTE_SINGLE_UNIT:
            want = _ROUTE_TABLE_CAP_32_BELOW_FILL.get(group, {}).get(s_q, want)
            cf = decode_route(geom, s_q, policy=policy, max_q_tile=32, batch=fill)
            want_f = _ROUTE_TABLE_CAP_32[policy].get(group, {}).get(s_q, _FILLED_ROUTES.get(group, {}).get(s_q, (tile, n_q, units, packed)))
            assert (cf.tile, cf.n_q, cf.token_units, cf.pack_gqa, cf.q_tile_max, cf.batch) == want_f + (32, fill), (group, s_q, cf)
        assert (c.tile, c.n_q, c.token_units, c.pack_gqa, c.q_tile_max) == want + (32,), (group, s_q, policy, c)
        assert c.n_q <= 32 and (c.claimed is (c.decode_tile and c.pack_gqa and s_q <= DECODE_CLAIMED_MAX_S_Q))
    with pytest.raises(ValueError, match="decode_q_tile_max must be one of"):
        decode_route(geom, 1, policy=policy, max_q_tile=48)
    mha = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=4, h_kv=4, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    r = decode_route(mha, 2, policy=policy)
    assert (r.group, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == (1, 16, 1, False, 1)
    d128 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=128, rope_dim=_ROPE, causal_bottom_right=True)
    assert not decode_route(d128, 1).served and "d_head=128" in decode_route(d128, 1).reason  # no decode tile outside the d256 flavor


@pytest.mark.parametrize("policy", [DECODE_ROUTE_SINGLE_UNIT, DECODE_ROUTE_TOKEN_UNITS])
@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_the_sm100_route_table_is_the_adapters_rule(group, policy, monkeypatch):
    """The DECODE route on the SM100 line (``decode_route(device_cc=(10, 0))``, 2026-10-29) at every served group under BOTH policies
    (they coincide there: the line has no token-unit arm): the table ``_SM100_ROUTE_TABLE`` cell for cell at the small-launch reading --
    the 16-column tile packed, the paged prefill tile packed at every batch for a group dividing 128, the 16-column tile unpacked past
    them, the typed decline of an unpacked step past 16 tokens (the 32-column arm), 33 the EXTEND form's --, the wide cells of
    ``_SM100_WIDE`` at the (batch, max_kv_len) pairs that make the launch wide (the arm: declined by name today, the adapter's 32-column
    tile once ``DECODE_WIDE_TILE_CCS`` names the part), ``claimed`` False on every cell, ``pdl_capable`` on the decode tile, the two
    readbacks, the cap refused with the adapter's reason; an UNSERVED part (cc 10.3, sm80) reads the Rubin table -- the home reading,
    ``device_cc`` None -- as the envelope and the declaration do (one rule), and the Rubin reading is the Rubin table."""
    geom, _ = _geom(group)
    for s_q, want in _SM100_ROUTE_TABLE[group].items():
        r = decode_route(geom, s_q, policy=policy, device_cc=_SM100)
        assert isinstance(r, DecodeRoute) and (r.s_q, r.group, r.policy, r.q_tile_max, r.device_cc, r.max_kv_len, r.batch) == (
            s_q,
            group,
            policy,
            None,
            _SM100,
            None,
            None,
        )
        if want is None:
            assert not r.served and r.tile is None and r.n_q == 0 and r.kv_streams == 0 and not r.claimed and not r.pdl_capable, (group, s_q, r)
            assert ("EXTEND form" in r.reason) if s_q > DECODE_MAX_S_Q else ("WIDE launches only" in r.reason and _WIDE_ARM_LIFT in r.reason), (group, s_q, r)
            continue
        assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == want, (group, s_q, policy, r)
        assert r.served and r.reason == "" and r.claimed is False and r.pdl_capable is (want[0] == _T), (group, s_q, r)
        assert r.decode_tile is (want[0] == _T) and r.prefill_tile is (want[0] == _P)
        # the line is batch-blind below the wide rule: a small launch at any batch is the table's cell (the prefill tile needs no fill)
        small = decode_route(geom, s_q, policy=policy, device_cc=_SM100, batch=1, max_kv_len=1024)
        assert (small.tile, small.n_q, small.token_units, small.pack_gqa, small.kv_streams) == want and small.batch == 1 and small.max_kv_len == 1024
        tu = decode_route(geom, s_q, policy=DECODE_ROUTE_TOKEN_UNITS, device_cc=_SM100)
        assert (
            tu.tile,
            tu.n_q,
            tu.token_units,
            tu.pack_gqa,
            tu.kv_streams,
        ) == want and tu.policy == DECODE_ROUTE_TOKEN_UNITS, "both policies coincide on the line"
        # an UNSERVED part (cc 10.3 until its cells admit it; an sm80 host) reads the Rubin table, the home reading, and says so
        # (``device_cc`` None) -- the same reading the envelope's routes / route() and the declaration take there; the Rubin reading is the
        # Rubin table under its own part
        for other in ((10, 3), (8, 0)):
            unserved = decode_route(geom, s_q, policy=policy, device_cc=other)
            assert unserved == decode_route(geom, s_q, policy=policy) and unserved.device_cc is None, (other, s_q, unserved)
        rubin = decode_route(geom, s_q, policy=policy, device_cc=_SM107)
        assert rubin == decode_route(geom, s_q, policy=policy) and rubin.device_cc == _SM107
    if group == 6:
        for (b, kv, s_q), (want, flipped) in _SM100_WIDE.items():
            r = decode_route(geom, s_q, policy=policy, device_cc=_SM100, batch=b, max_kv_len=kv)
            got = None if not r.served else (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams)
            assert got == want and (r.batch, r.max_kv_len) == (b, kv) and not r.claimed, (b, kv, s_q, r)
            if want is None:
                # the arm's reason names the lift on both readings; a wide launch names itself, a small one the rule it is off
                assert _WIDE_ARM_LIFT in r.reason and ("at a WIDE launch" in r.reason) is (flipped is not None), (b, kv, s_q, r.reason)
            # the one-tuple lift: with the part in DECODE_WIDE_TILE_CCS the same inputs read the adapter's 32-column arm; nothing else moves
            with monkeypatch.context() as m:
                m.setattr(_api, "DECODE_WIDE_TILE_CCS", (_SM100,))
                f = decode_route(geom, s_q, policy=policy, device_cc=_SM100, batch=b, max_kv_len=kv)
                assert (None if not f.served else (f.tile, f.n_q, f.token_units, f.pack_gqa, f.kv_streams)) == flipped, (b, kv, s_q, f)
    # the cap names the Rubin table: not served on the line with the adapter's own reason, under either policy
    capped = decode_route(geom, 1, policy=policy, device_cc=_SM100, max_q_tile=32)
    assert not capped.served and "decode_q_tile_max=32 caps the Rubin" in capped.reason and "decode_q_tile_max=None" in capped.reason
    with pytest.raises(ValueError, match="max_kv_len must be >= 1"):
        decode_route(geom, 1, device_cc=_SM100, max_kv_len=0)
    mha = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=4, h_kv=4, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    r = decode_route(mha, 16, policy=policy, device_cc=_SM100)
    assert (r.group, r.n_q, r.token_units, r.pack_gqa, r.kv_streams, r.claimed) == (1, 16, 1, False, 1, False)
    # 17 tokens unpacked = the 32-column arm: declined at a small launch (off the rule) and at a wide one (the arm), the lift named on both
    small17 = decode_route(mha, 17, policy=policy, device_cc=_SM100)
    wide17 = decode_route(mha, 17, policy=policy, device_cc=_SM100, batch=8, max_kv_len=65536)
    assert not small17.served and not wide17.served and _WIDE_ARM_LIFT in small17.reason and _WIDE_ARM_LIFT in wide17.reason
    assert "WIDE launches only" in small17.reason and "at a WIDE launch" in wide17.reason and "batch 8 x H_q 4 = 32 units" in wide17.reason


def test_the_32_column_sm100_decode_arm_is_declined_by_name_until_its_cells_run(monkeypatch):
    """The cc 10.0 admission's review follow-up (2026-10-29, M1): the SDPA adapter's rule off cc 10.7 takes the 32-column decode tile for
    17 .. 32 packed rows and for an unpacked step of 17 .. 32 tokens on a WIDE launch, but no cc 10.0 cell of the block has run that arm,
    so ``decode_route`` declines it by name on every part off ``DECODE_WIDE_TILE_CCS`` (EMPTY today) with the lift in the reason -- at the
    geometries that would have hit it: W at group 6 (B 32 x kv 4K, S_q 3), group 16 (B 8 x kv 32K, S_q 2: 32 packed rows) and group 4
    (B 8 x kv 64K, S_q 8); U32 at group 6 (B 4 x kv 5.5K, S_q 17) and MHA (B 8 x kv 16K, S_q 17).  The small-launch reading of the same
    steps is unchanged, the envelope predicts the verdict at the same inputs, the Rubin reading never meets the gate, and the one-tuple
    lift restores the adapter's rule (the plumbing is intact): with ``(10, 0)`` in the tuple the same inputs read the 32-column tile,
    packed or unpacked, in one unit.  A host row: ``decode_route`` is device-free."""
    assert DECODE_WIDE_TILE_CCS == () and _api.DECODE_WIDE_TILE_CCS == ()
    mha = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=4, h_kv=4, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    cells = [  # (geometry, S_q, batch, max_kv_len, the arm's reading once admitted, the small-launch reading of the same step today)
        (_geom(6)[0], 3, 32, 4096, _W32, _U(6)),
        (_geom(16)[0], 2, 8, 32768, _W32, (_P, 0, 0, True, 1)),
        (_geom(4)[0], 8, 8, 65536, _W32, (_P, 0, 0, True, 1)),
        (_geom(6)[0], 17, 4, 5632, _U32, None),
        (mha, 17, 8, 16384, (_T, 32, 1, False, 1), None),
    ]
    for geom, s_q, b, kv, arm, small in cells:
        for policy in DECODE_ROUTE_POLICIES:
            r = decode_route(geom, s_q, policy=policy, device_cc=_SM100, batch=b, max_kv_len=kv)
            assert not r.served and r.tile is None and r.n_q == 0 and r.kv_streams == 0 and not r.claimed and not r.pdl_capable, (s_q, b, kv, r)
            assert _WIDE_ARM_LIFT in r.reason and "at a WIDE launch" in r.reason and f"batch {b} x" in r.reason and f"max_kv_len={kv}" in r.reason, r.reason
            assert (r.device_cc, r.batch, r.max_kv_len, r.policy) == (_SM100, b, kv, policy)
            s = decode_route(geom, s_q, policy=policy, device_cc=_SM100, batch=1, max_kv_len=1024)
            assert (None if not s.served else (s.tile, s.n_q, s.token_units, s.pack_gqa, s.kv_streams)) == small, (s_q, s)
            if small is None:
                assert _WIDE_ARM_LIFT in s.reason and "WIDE launches only" in s.reason and "at a WIDE launch" not in s.reason
        env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM100).decode
        assert (
            env.served
            and env.route_cc == _SM100
            and env.route(s_q, batch=b, max_kv_len=kv) == decode_route(geom, s_q, device_cc=_SM100, batch=b, max_kv_len=kv)
        )
        assert not env.route(s_q, batch=b, max_kv_len=kv).served and _WIDE_ARM_LIFT in env.route(s_q, batch=b, max_kv_len=kv).reason
        rubin = decode_route(geom, s_q, device_cc=_SM107, batch=b, max_kv_len=kv)  # cc 10.7 reads its own table at the same inputs
        assert rubin == decode_route(geom, s_q, batch=b) and rubin.device_cc == _SM107 and rubin.served
        with monkeypatch.context() as m:
            m.setattr(_api, "DECODE_WIDE_TILE_CCS", (_SM100,))
            f = decode_route(geom, s_q, device_cc=_SM100, batch=b, max_kv_len=kv)
            assert f.served and (f.tile, f.n_q, f.token_units, f.pack_gqa, f.kv_streams) == arm and f.reason == "" and not f.claimed and f.pdl_capable, (s_q, f)
            assert GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM100).decode.route(s_q, batch=b, max_kv_len=kv) == f


def test_the_split_choice_is_the_decode_split_model_floored_at_two_under_the_gate():
    """The block's split is the SDPA decode split model's on the same inputs -- the launch's units, KV tiles, SM count, the
    tile's width and the COMBINE GRID (``combine_grid`` = (B, H_q, S_q): the lean combine kernel's ceil(S_q / 4) x H_q x B blocks, which
    decides the form the combine launch takes and so its price per partial) -- floored at 2 under the gate.  A block at batch 1
    on a long cache is where the grid matters (the decode-grid combine's flat price lets the model split deeper than the lean
    kernel's walk would pay for): pinned against the model with and without the grid."""
    from cudnn.sdpa.fwd.heuristics import choose_decode_tile_split_kv

    geom, _ = _geom(16)
    r = decode_route(geom, 2)
    for sm in (108, 148, 204, 212):
        want = choose_decode_tile_split_kv(units=4 * 2 * r.token_units, kv_tiles=8, sm_count=sm, q_tile=r.n_q, combine_grid=(4, 32, 2))
        assert decode_split_choice(r, batch=4, h_kv=2, max_kv_len=1000, sm_count=sm, gated=False, h_q=32) == want
        assert decode_split_choice(r, batch=4, h_kv=2, max_kv_len=1000, sm_count=sm, gated=True, h_q=32) == max(want, 2)
    t = decode_route(geom, 5, policy=DECODE_ROUTE_TOKEN_UNITS)  # the token-unit arm: two units of the 64-column tile, each a stream
    assert t.pack_gqa and t.decode_tile and (t.n_q, t.token_units) == (64, 2)
    want_t = choose_decode_tile_split_kv(units=4 * 2 * 2, kv_tiles=8, sm_count=204, q_tile=64, combine_grid=(4, 32, 5), token_units=2)
    assert decode_split_choice(t, batch=4, h_kv=2, max_kv_len=1000, sm_count=204, gated=False, h_q=32) == want_t
    odd = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=34, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    u = decode_route(odd, 7, policy=DECODE_ROUTE_TOKEN_UNITS)  # group 17 at S_q 7 = 119 rows: three units of 64, so the UNPACKED arm, one Q head per CTA
    assert not u.pack_gqa and u.decode_tile and (u.n_q, u.token_units, u.kv_streams) == (16, 1, 17)
    want_u = choose_decode_tile_split_kv(units=4 * 34, kv_tiles=8, sm_count=204, q_tile=16, combine_grid=(4, 34, 7))
    assert decode_split_choice(u, batch=4, h_kv=2, max_kv_len=1000, sm_count=204, gated=False, h_q=34) == want_u
    # batch 1 on a 32K cache at 212 SMs: the block's launch is a 32-block lean grid, so the combine runs on the decode-grid kernel
    # and the model's split is the grid-aware one (deeper than the lean price allows); without h_q the group x h_kv stands in
    one = decode_route(geom, 1)
    with_grid = choose_decode_tile_split_kv(units=2, kv_tiles=256, sm_count=212, q_tile=16, combine_grid=(1, 32, 1))
    without = choose_decode_tile_split_kv(units=2, kv_tiles=256, sm_count=212, q_tile=16)
    assert with_grid > without and decode_split_choice(one, batch=1, h_kv=2, max_kv_len=32768, sm_count=212, gated=False, h_q=32) == with_grid
    assert decode_split_choice(one, batch=1, h_kv=2, max_kv_len=32768, sm_count=212, gated=False) == with_grid
    p = decode_route(geom, 5, batch=32)  # the prefill route (at the fill): the paged THD body runs unsplit
    assert p.prefill_tile and decode_split_choice(p, batch=32, h_kv=2, max_kv_len=1000, sm_count=204, gated=True, h_q=32) == 1
    assert decode_split_choice(decode_route(geom, 33), batch=4, h_kv=2, max_kv_len=1000, sm_count=204, gated=True, h_q=32) == 1  # an unserved route
    assert decode_split_choice(r, batch=4, h_kv=2, max_kv_len=1000, sm_count=0, gated=True) == 1  # no device: declined later anyway


def test_the_prefill_route_needs_the_step_to_fill_the_part(monkeypatch):
    _rubin_reading(monkeypatch)
    """The DECODE form's prefill route is ONE unsplit CTA per (sequence, KV head), so below ``DECODE_PREFILL_ROUTE_MIN_CTAS`` = 64 CTAs
    (``batch x H_kv``) it leaves the part idle while the decode tile's two-unit arm splits the same KV over it.  Measured on the Rubin
    performance class (212 SMs) at 32 / 2, ``S_q = 8``, kv 32K: B 1 took 330.6 us on the prefill route against 58.7 on two 64-column
    units; B 32 (64 CTAs) 389.9 on the prefill route against 472.6 on the two units.  So the default policy takes the prefill tile only
    where the step fills the part (or where no two-unit decode route holds the step), and reads the batch back: 32 / 2 at ``S_q`` 5 .. 8
    rides two 64-column units at B 1 and B 4 and the prefill tile at B 32, on the route record, the declared block and the envelope;
    ``token_units`` is the two-unit route at every batch; a cap that leaves no two-unit route (``decode_q_tile_max=16`` for 4 / 1 at
    ``S_q`` 17) keeps the prefill tile at any batch; a batch below 1 is a typed ``ValueError``."""
    geom, _ = _geom(16)
    for s_q in (5, 6, 7, 8):
        for b in (1, 4):
            r = decode_route(geom, s_q, batch=b)
            assert r.decode_tile and r.pack_gqa and (r.n_q, r.token_units, r.kv_streams, r.batch, r.pdl_capable) == (64, 2, 2, b, True), (s_q, b, r)
            t = decode_route(geom, s_q, policy=DECODE_ROUTE_TOKEN_UNITS)  # the same kernel the token-unit policy names, under the default policy
            assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == (
                t.tile,
                t.n_q,
                t.token_units,
                t.pack_gqa,
                t.kv_streams,
            ) and r.policy == DECODE_ROUTE_SINGLE_UNIT
        f = decode_route(geom, s_q, batch=32)
        assert f.prefill_tile and (f.n_q, f.token_units, f.kv_streams, f.batch, f.pdl_capable) == (0, 0, 1, 32, False), (s_q, f)
        assert decode_route(geom, s_q, batch=31) == decode_route(geom, s_q, batch=1) and decode_route(geom, s_q, batch=128).prefill_tile
        assert decode_route(geom, s_q, policy=DECODE_ROUTE_TOKEN_UNITS, batch=32).decode_tile
        for b in (1, 4, 32):
            blk, _ = _make_block(16, b, s_q, fuse_gate=True)
            assert blk.decode_route == decode_route(geom, s_q, batch=b) and blk.decode_route.batch == b, (s_q, b, blk.decode_route)
            if b < 32:
                assert blk._sdpa.paged_dense and not blk._sdpa.thd and blk.decode_split_kv >= 2 and blk.gate_fused, (s_q, b)
            else:
                assert blk._sdpa.thd and blk.decode_split_kv == 1 and blk.gate_fused, (s_q, b)
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
    assert env.decode.route(8).decode_tile and env.decode.route(8).token_units == 2 and env.decode.route(8, batch=32).prefill_tile
    assert env.decode.route(8, batch=4).decode_tile and env.decode.route(8).batch is None and env.decode.route(8, batch=32).batch == 32
    assert [x.tile for x in env.decode.routes][4:8] == [DECODE_TILE] * 4, "the envelope's routes are the small-batch reading"
    # the fill is a CTA count: one KV head needs 64 sequences, two need 32; MHA and the other groups' single-unit cells do not move
    g16_1 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=16, h_kv=1, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    assert decode_route(g16_1, 8, batch=32).decode_tile and decode_route(g16_1, 8, batch=64).prefill_tile
    for group in sorted(_GROUPS):
        gm, _ = _geom(group)
        for s_q in range(1, 5):
            assert decode_route(gm, s_q, batch=1) == decode_route(gm, s_q, batch=4) == decode_route(gm, s_q, batch=128) == decode_route(gm, s_q), (group, s_q)
    g4, _ = _geom(4)
    for b in (1, 64):
        c = decode_route(g4, 17, max_q_tile=16, batch=b)
        assert c.prefill_tile and c.batch == b, c  # no two-unit route under the cap: the prefill tile whatever the batch
    with pytest.raises(ValueError, match="batch must be >= 1"):
        decode_route(geom, 8, batch=0)


def test_the_two_unit_routes_split_choice_prices_one_kv_stream_per_split():
    """The two token units of one (sequence, KV head) stream the SAME KV range, so the block hands the model the route's token
    units and the model counts one KV stream per split for the pair.  Measured on the Rubin performance class (212 SMs) at
    B 32 x kv 32K, S_q 4 on the two-unit 32-column route (the width cap's arm at 24 / 2): 219.4 us at split 3 against 277.6
    unsplit, 249.1 at 4 and 307.8 at 2 -- the block's choice is 3, where the same 128 CTAs priced as single units read unsplit.
    The one-unit 64-column route of the same step keeps its one-wave split 3 (the single-unit pricing)."""
    from cudnn.sdpa.fwd.heuristics import choose_decode_tile_split_kv

    geom, _ = _geom(12)
    r = decode_route(geom, 4, max_q_tile=32)
    assert r.decode_tile and r.pack_gqa and (r.n_q, r.token_units) == (32, 2)
    got = decode_split_choice(r, batch=32, h_kv=2, max_kv_len=32768, sm_count=212, gated=False, h_q=24)
    assert got == 3, got
    assert got == choose_decode_tile_split_kv(units=32 * 2 * 2, kv_tiles=256, sm_count=212, q_tile=32, combine_grid=(32, 24, 4), token_units=2)
    assert decode_split_choice(r, batch=32, h_kv=2, max_kv_len=32768, sm_count=212, gated=True, h_q=24) == 3
    r64 = decode_route(geom, 4)
    assert (r64.n_q, r64.token_units) == (64, 1) and decode_split_choice(r64, batch=32, h_kv=2, max_kv_len=32768, sm_count=212, gated=False, h_q=24) == 3
    # the 24 / 2 MTP step at S_q 8 (two units of the 64-column tile) and the 32 / 2 one under ``token_units``: the model's own pick
    # on the shared streams, equal to the model called with the route's units
    for g, s_q, policy in ((12, 8, None), (16, 8, DECODE_ROUTE_TOKEN_UNITS)):
        gm, _ = _geom(g)
        rt = decode_route(gm, s_q, policy=policy)
        assert (rt.n_q, rt.token_units) == (64, 2), rt
        h_q = _GROUPS[g][0]
        want = choose_decode_tile_split_kv(units=32 * 2 * 2, kv_tiles=256, sm_count=212, q_tile=64, combine_grid=(32, h_q, s_q), token_units=2)
        assert decode_split_choice(rt, batch=32, h_kv=2, max_kv_len=32768, sm_count=212, gated=False, h_q=h_q) == want


def test_the_envelope_on_a_rubin_reading_serves_the_decode_form_and_names_every_route(monkeypatch):
    from cudnn.frost import buffers

    # This is a hypothetical supported Rubin declaration, including its DSL target.
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom, _ = _geom(12)
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
    assert isinstance(env, ServingEnvelope) and isinstance(env.decode, DecodeEnvelope) and env.cutedsl_ok and env.device_cc == _SM107
    d = env.decode
    assert d.served and d.reason is None and env.extend.served and env.forms == ("extend", "decode")
    assert d.page_sizes == SERVING_PAGE_SIZES and d.pool_layouts == SERVING_POOL_LAYOUTS and d.max_s_q == DECODE_MAX_S_Q and d.max_rows == DECODE_MAX_ROWS
    assert d.dtypes == (torch.bfloat16, torch.float16) and d.inference_only and not d.writes_lse and d.requires
    assert (
        [r.s_q for r in d.routes] == list(range(1, DECODE_MAX_S_Q + 2)) and d.claimed_max_s_q == DECODE_CLAIMED_MAX_S_Q and d.policy == DECODE_ROUTE_SINGLE_UNIT
    )
    # 24 / 2: the prefill tile cannot pack a 12-head group, so the decode tile serves every S_q <= 32: PACKED to S_q 10 since the 64-column
    # width landed (one unit at S_q 3 .. 5, two units of it at 6 .. 10; the former unpacked arm past S_q 4 is the decode_q_tile_max=32 route),
    # UNPACKED (12 streams) from S_q 11 to 32 -- the 16-column tile to 16 tokens, the 32-column tile past it -- and declined at 33
    assert [r.tile for r in d.routes] == [DECODE_TILE] * 32 + [None] and [r.pack_gqa for r in d.routes] == [True] * 10 + [False] * 23
    assert [r.n_q for r in d.routes] == [16, 32, 64, 64, 64, 64, 64, 64, 64, 64] + [16] * 6 + [32] * 16 + [0]
    assert [r.token_units for r in d.routes] == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] + [1] * 22 + [0]
    assert [r.kv_streams for r in d.routes] == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] + [12] * 22 + [0]
    assert d.route(3).token_units == 1 and d.route(6).token_units == 2 and d.route(8).served and d.route(32).served and not d.route(33).served
    assert "EXTEND form" in d.route(33).reason
    capped = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_q_tile_max=32).decode
    assert capped.q_tile_max == 32 and [r.pack_gqa for r in capped.routes] == [True] * 4 + [False] * 29 and capped.route(3).token_units == 2
    assert all(r.served for r in capped.routes[:-1]) and not capped.routes[-1].served
    # the programmatic-dependent-launch readbacks: the switched stage kinds per form, stage (4) on a decode-tile route only
    assert d.pdl_stages == ("qk_norm_rope", "cache_write", "sigmoid_gate", "sdpa") and env.extend.pdl_stages == ("qk_norm_rope", "cache_write", "sigmoid_gate")
    assert all(r.pdl_capable is r.decode_tile for r in d.routes) and d.route(1).pdl_capable and not d.route(33).pdl_capable
    # the AUTO reading of the switch on this part (a Rubin reading of a SERVED form): on; off on every other part and on an unserved dtype
    # (a form-level reading: the per-step one is `pdl_default and route(s_q, batch).pdl_capable`)
    assert d.pdl_default is True and GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=(10, 0)).decode.pdl_default is False
    fp32 = GatedAttentionBlockFwd.serving_envelope(geom, torch.float32, _DEV, device_cc=_SM107).decode
    assert not fp32.served and fp32.pdl_default is False, "the default is a reading of a served form"
    assert (d.pdl_default and d.route(1).pdl_capable) is True and (d.pdl_default and d.route(33).pdl_capable) is False
    g16, _ = _geom(16)
    e16 = GatedAttentionBlockFwd.serving_envelope(g16, torch.bfloat16, _DEV, device_cc=_SM107).decode
    assert e16.route(4).pdl_capable and not e16.route(5, batch=32).pdl_capable and e16.route(5, batch=32).prefill_tile and e16.route(9).pdl_capable
    assert e16.route(5).pdl_capable and e16.route(5).decode_tile, "below the prefill route's fill the step rides the decode tile"
    # (the table's `sparse` row is the PACKED form's since 2026-10-14: a dense QsaSpec declaration over the pools is the block-sparse decode mode)
    assert set(d.declines) == {"write_through", "decode_rows", "diagonal", "window", "head_dim", "return_lse", "token_capacity_dense"}
    f16 = GatedAttentionBlockFwd.serving_envelope(geom, torch.float16, _DEV, device_cc=_SM107)
    assert f16.decode.served and f16.extend.served


@pytest.mark.parametrize("version, has_target", [("4.7.0", False), ("4.8.0", False), ("4.8.0", True)])
def test_the_envelope_requires_the_rubin_dsl_target(monkeypatch, version, has_target):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", version)))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: has_target)
    geom, _ = _geom(16)
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, "cpu", device_cc=_SM107)
    assert env.cutedsl_ok is has_target
    assert env.extend.served is has_target and env.decode.served is has_target
    assert env.decode.pdl_default is has_target  # the default follows the served form: no Rubin target, no PDL default
    assert env.forms == (("extend", "decode") if has_target else ())
    assert env.decode.route(1).served  # Geometry remains inspectable without a runnable target.
    if has_target:
        assert env.cutedsl_reason is None and env.extend.reason is None and env.decode.reason is None
    else:
        assert version in env.cutedsl_reason and "sm_107a" in env.cutedsl_reason
        assert env.extend.reason == env.decode.reason == env.cutedsl_reason


@pytest.mark.parametrize(
    ("cc", "match"),
    [((8, 0), "found SM80"), ((10, 3), "found SM103"), ((12, 0), "found SM120"), (None, "no CUDA device")],
    ids=["sm80", "sm103", "sm120", "none"],
)
def test_the_envelope_declines_every_other_part_without_raising(cc, match):
    """Every part off ``DENSE_SERVING_CCS`` -- cc 10.3 among them: the same SM100 kernels, unproven through the block, declined BY NAME
    until its cells run -- reads not served with the part statement on both forms; the routes stay listed (geometry facts)."""
    geom, _ = _geom(16)
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, "cpu" if cc is None else _DEV, device_cc=cc)
    assert not env.decode.served and not env.extend.served and env.forms == () and match in env.decode.reason and match in env.extend.reason
    assert "served on cc 10.7 (Rubin, SM107: every form) and cc 10.0" in env.decode.reason and env.decode.reason == env.extend.reason
    if cc == (10, 3):
        assert "cc 10.3 runs the same SM100 kernels but is unproven through the block: declined by name" in env.decode.reason
    assert env.decode.routes  # the routes are geometry facts, listed either way
    assert env.decode.pdl_default is False  # the PDL default is the measured part's
    assert cc not in DENSE_SERVING_CCS and DENSE_SERVING_CCS == (_SM107, _SM100)
    # (2026-10-29, L5) an UNSERVED part reads the Rubin table on every reading -- the envelope's routes / route() and
    # decode_route(device_cc=) agree (one rule, _route_part) and both say which table they show (route_cc / device_cc None = the Rubin
    # table, the home reading)
    assert env.decode.route_cc is None and all(r.device_cc is None for r in env.decode.routes)
    home = tuple(decode_route(geom, s) for s in range(1, DECODE_MAX_S_Q + 2))
    assert env.decode.routes == tuple(decode_route(geom, s, device_cc=cc) for s in range(1, DECODE_MAX_S_Q + 2)) == home
    at_launch = env.decode.route(2, batch=64, max_kv_len=65536)
    assert at_launch == decode_route(geom, 2, device_cc=cc, batch=64, max_kv_len=65536) == decode_route(geom, 2, batch=64)
    assert at_launch.device_cc is None and at_launch.n_q == 32 and at_launch.pack_gqa, at_launch  # the Rubin table's 32-column single unit


def test_the_envelope_on_a_cc_10_0_reading_serves_the_dense_forms_and_names_the_sm100_routes():
    """The cc 10.0 reading (2026-10-29): both dense forms served for bf16 / f16, the DECODE routes the SM100 line's (the table above:
    ``_SM100_ROUTE_TABLE``), ``route(s_q, batch, max_kv_len)`` reading the wide rule, the PDL default off (``DECODE_PDL_DEFAULT_CCS`` is
    the measured part's), ``claimed`` False on every route, the cap refused in the routes, the declines table the dense form's, a
    QsaSpec geometry NOT served there (the sparse forms stay Rubin-only through the block), fp32 declined on the dtype row."""
    for group in sorted(_GROUPS):
        geom, _ = _geom(group)
        env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM100)
        assert env.forms == ("extend", "decode") and env.extend.served and env.decode.served and env.device_cc == _SM100, (group, env.decode.reason)
        assert env.extend.reason is None and env.decode.reason is None and env.cutedsl_ok
        d = env.decode
        assert d.pdl_default is False and d.policy == DECODE_ROUTE_SINGLE_UNIT and d.q_tile_max is None and d.claimed_max_s_q == DECODE_CLAIMED_MAX_S_Q
        assert [r.s_q for r in d.routes] == list(range(1, DECODE_MAX_S_Q + 2)) and all(r.device_cc == _SM100 and not r.claimed for r in d.routes)
        for r in d.routes:
            want = _SM100_ROUTE_TABLE[group][r.s_q]
            got = None if not r.served else (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams)
            assert got == want, (group, r.s_q, r)
            assert d.route(r.s_q) == r and d.route(r.s_q).device_cc == _SM100
        assert all(r.pdl_capable is r.decode_tile for r in d.routes)
        assert set(d.declines) == {"write_through", "decode_rows", "diagonal", "window", "head_dim", "return_lse", "token_capacity_dense"}
        assert "a part of DENSE_SERVING_CCS" in " ".join(d.requires) and "a part of DENSE_SERVING_CCS" in " ".join(env.extend.requires)
        assert GatedAttentionBlockFwd.serving_envelope(geom, torch.float16, _DEV, device_cc=_SM100).forms == ("extend", "decode")
        fp32 = GatedAttentionBlockFwd.serving_envelope(geom, torch.float32, _DEV, device_cc=_SM100)
        assert not fp32.decode.served and "bf16 / f16" in fp32.decode.reason and fp32.decode.pdl_default is False
    g6, _ = _geom(6)
    e6 = GatedAttentionBlockFwd.serving_envelope(g6, torch.bfloat16, _DEV, device_cc=_SM100).decode
    assert e6.route(3).decode_tile and not e6.route(3).pack_gqa and e6.route(3).kv_streams == 6, "the small-launch reading: U16"
    w = e6.route(3, batch=32, max_kv_len=4096)  # the wide reading: the adapter's 32-column arm = a typed decline naming the lift (M1)
    assert not w.served and _WIDE_ARM_LIFT in w.reason and "at a WIDE launch" in w.reason and (w.batch, w.max_kv_len) == (32, 4096), w
    u = e6.route(17, batch=4, max_kv_len=5632)
    assert not e6.route(17).served and not u.served and _WIDE_ARM_LIFT in u.reason and "at a WIDE launch" in u.reason
    assert u.max_kv_len == 5632 and u.batch == 4 and u.device_cc == _SM100
    capped = GatedAttentionBlockFwd.serving_envelope(g6, torch.bfloat16, _DEV, device_cc=_SM100, decode_q_tile_max=32).decode
    assert capped.q_tile_max == 32 and not any(r.served for r in capped.routes) and "decode_q_tile_max=32 caps the Rubin" in capped.route(1).reason
    # the sparse forms: Rubin-only through the block (the standalone adapter's explicit lean tile on cc 10.0 is not the block's)
    sparse = GatedAttentionBlockFwd.serving_envelope(
        GatedAttentionBlockGeometry(**dict(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True, qsa=QsaSpec())),
        torch.bfloat16,
        _DEV,
        device_cc=_SM100,
    )
    assert (
        sparse.forms == () and not sparse.decode.served and "sparse forms" in sparse.decode.reason and "served on cc 10.7 (Rubin) only" in sparse.decode.reason
    )
    assert sparse.decode.routes == () and sparse.decode.pdl_default is False


def test_every_static_decline_of_the_decode_declaration_is_predicted_by_the_envelope(monkeypatch):
    """The contract a serving stack relies on: no decline the DECODE declaration raises for a (geometry, dtype) that the
    envelope did not predict.  Each row: the envelope on a Rubin reading names the reason AND the declaration raises it."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom16, _ = _geom(16)
    d128 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=128, rope_dim=_ROPE, causal_bottom_right=True)
    # a TOP-LEFT QsaSpec geometry meets the `diagonal` row like a dense one (a ValueError); with the bottom-right diagonal it is the
    # block-sparse DECODE mode, SERVED since 2026-10-14 (test_block_qsa_decode.py) -- the former `sparse` row is the packed form's
    sparse = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, qsa=QsaSpec())
    rows = [
        (geom16, torch.float8_e4m3fn, "bf16 / f16", NotImplementedError),
        (d128, torch.bfloat16, "d_head == 256", NotImplementedError),
        (sparse, torch.bfloat16, "causal_bottom_right=True", ValueError),
    ]
    for geom, dtype, match, _exc in rows:
        env = GatedAttentionBlockFwd.serving_envelope(geom, dtype, _DEV, device_cc=_SM107)
        assert not env.decode.served and match in env.decode.reason and "decode" not in env.forms, (match, env.decode.reason)
    served = GatedAttentionBlockFwd.serving_envelope(
        GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True, qsa=QsaSpec()),
        torch.bfloat16,
        _DEV,
        device_cc=_SM107,
    )
    assert served.decode.served and served.decode.sparse_core and served.forms == ("decode",) and served.decode.routes == ()
    # ... and the declaration raises the same row (device-free), the e4m3 one through the quant halves it needs first
    for geom, dtype, match, exc in rows[1:]:
        mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=_DEV)  # noqa: E731
        kc, vc = _pools(4, 2, 16, geom.d_head, "hnd", dtype)
        with pytest.raises(exc, match=match):
            GatedAttentionBlockFwd(
                mk(1, 1, _D_MODEL),
                mk(geom.n_qkvg, _D_MODEL),
                torch.ones(geom.d_head, dtype=dtype, device=_DEV),
                torch.ones(geom.d_head, dtype=dtype, device=_DEV),
                mk(1, 1, _ROPE),
                mk(1, 1, _ROPE),
                mk(_D_MODEL, 8 * geom.d_head),
                mk(1, 1, _D_MODEL),
                geom,
                paged_kv_page_size=16,
                kv_source="paged",
                sample_k_cache=kc,
                sample_v_cache=vc,
                sample_block_table=torch.zeros(1, 4, dtype=torch.int32, device=_DEV),
            )
    # the device rows the SDPA stage raises at check_support are the envelope's device reasons (test_the_envelope_declines_every_other_part)
    assert GatedAttentionBlockFwd.serving_envelope(geom16, torch.bfloat16, _DEV).decode.served is (_api._device_cc_of(_DEV) in DENSE_SERVING_CCS)


# ---------------------------------------------------------------------------
# The declaration: rows, facts
# ---------------------------------------------------------------------------


def test_declaration_rows_are_typed_and_name_the_feature(monkeypatch):
    _rubin_reading(monkeypatch)
    kc, vc = _pools(4, 2, 16, _D, "hnd", torch.bfloat16)
    bt = torch.zeros(2, 4, dtype=torch.int32, device=_DEV)
    with pytest.raises(ValueError, match="kv_source must be one of"):
        _make_block(16, 2, 1, kv_source="bogus")
    with pytest.raises(ValueError, match="kv_source must be one of"):
        _make_block(16, 2, 1, kv_source=True)
    with pytest.raises(ValueError, match="needs paged_kv_page_size > 0"):
        _make_block(16, 2, 1, paged_kv_page_size=0)
    with pytest.raises(ValueError, match="belong to the EXTEND form or the DECODE form"):
        _make_block(16, 2, 1, kv_source="self", paged_kv_page_size=16)
    for missing in ("sample_k_cache", "sample_v_cache", "sample_block_table"):
        with pytest.raises(ValueError, match=f"needs {missing} at declaration"):
            _make_block(16, 2, 1, **{missing: None})
    with pytest.raises(ValueError, match="needs geometry.is_causal=True and geometry.causal_bottom_right=True"):
        geom = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=32, h_kv=2, d_head=_D, rope_dim=_ROPE)  # top-left causal
        mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
        GatedAttentionBlockFwd(
            mk(2, 1, _D_MODEL),
            mk(geom.n_qkvg, _D_MODEL),
            mk(_D),
            mk(_D),
            mk(2, 1, _ROPE),
            mk(2, 1, _ROPE),
            mk(_D_MODEL, 32 * _D),
            mk(2, 1, _D_MODEL),
            geom,
            paged_kv_page_size=16,
            kv_source="paged",
            sample_k_cache=kc,
            sample_v_cache=vc,
            sample_block_table=bt,
        )
    with pytest.raises(ValueError, match="sample_k_cache must be \\[num_pages >= 1, H=2, page_size=16, D=256\\]"):
        _make_block(16, 2, 1, sample_k_cache=torch.empty(4, 2, 32, _D, dtype=torch.bfloat16, device=_DEV))
    with pytest.raises(ValueError, match="same \\[num_pages, H_kv, P, D\\] shape"):
        _make_block(16, 2, 1, sample_v_cache=torch.empty(8, 2, 16, _D, dtype=torch.bfloat16, device=_DEV))
    with pytest.raises(ValueError, match="one layout kind"):
        _make_block(
            16,
            2,
            1,
            sample_k_cache=torch.empty(64, 2, 16, _D, dtype=torch.bfloat16, device=_DEV),
            sample_v_cache=torch.empty(64, 16, 2, _D, dtype=torch.bfloat16, device=_DEV).permute(0, 2, 1, 3),
        )
    with pytest.raises(ValueError, match="2-D int32"):
        _make_block(16, 2, 1, sample_block_table=torch.zeros(2, 64, dtype=torch.int64, device=_DEV))
    with pytest.raises(ValueError, match="one row per sequence"):
        _make_block(16, 2, 1, sample_block_table=torch.zeros(3, 64, dtype=torch.int32, device=_DEV))
    with pytest.raises(ValueError, match="page axis must be contiguous"):
        _make_block(16, 2, 1, sample_block_table=torch.zeros(64, 2, dtype=torch.int32, device=_DEV).t())
    with pytest.raises(ValueError, match="exceeds the block table's reach"):
        _make_block(16, 2, 1, max_kv_len=64 * 16 + 1)
    with pytest.raises(ValueError, match="max_kv_len must be a positive int"):
        _make_block(16, 2, 1, max_kv_len=0)
    with pytest.raises(ValueError, match="decode_split_kv must be None"):
        _make_block(16, 2, 1, decode_split_kv=0)
    # past the served envelope (S_q 32: the unpacked arm's widest step) a dense step over the pools is the EXTEND form's packed business:
    # raised at declaration; S_q 8 rides the prefill tile at this group once the step fills the part (32 sequences) and two units of the
    # 64-column tile below it, S_q 9 .. 32 the decode tile unpacked (16 streams)
    with pytest.raises(NotImplementedError, match=f"serves S_q <= {DECODE_MAX_S_Q}.*EXTEND form"):
        _make_block(16, 2, DECODE_MAX_S_Q + 1)
    with pytest.raises(NotImplementedError, match=f"serves S_q <= {DECODE_MAX_S_Q}.*EXTEND form"):
        _make_block(16, 2, 100)
    blk8, _ = _make_block(16, 32, 8)
    assert blk8.decode_route.served and blk8.decode_route.prefill_tile and blk8.decode_route.pack_gqa and blk8.decode_route.kv_streams == 1
    blk8b, _ = _make_block(16, 2, 8)
    assert blk8b.decode_route.decode_tile and blk8b.decode_route.pack_gqa and (blk8b.decode_route.token_units, blk8b.decode_route.kv_streams) == (2, 2)
    blk32, _ = _make_block(16, 2, DECODE_MAX_S_Q)
    assert blk32.decode_route.served and blk32.decode_route.decode_tile and not blk32.decode_route.pack_gqa and blk32.decode_route.kv_streams == 16
    assert blk32.decode_route.n_q == 32 and _make_block(16, 2, 9)[0].decode_route.n_q == 16
    # quant and save_for_backward with the attribute stay the write-through's own typed declines (named in the extend module)
    with pytest.raises(NotImplementedError, match="save_for_backward"):
        _make_block(16, 2, 1, save_for_backward=True, inplace_qkv=False, return_lse=True)
    # the rows of the shared serving table a dense paged declaration can hit: an LSE, a token capacity (THD-only)
    with pytest.raises(NotImplementedError, match="writes no LSE"):
        _make_block(16, 2, 1, return_lse=True)
    with pytest.raises(ValueError, match="token_capacity is THD-only"):
        _make_block(16, 2, 1, token_capacity=2)


def test_the_thd_product_bound_names_the_decode_form():
    """The ``S = 1`` wording of the packed envelope's product bound names the DECODE form (``S_q = 1`` with
    ``kv_source='paged'``), not a bare 'decode is out of scope'."""
    geom, _ = _geom(16)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    with pytest.raises(ValueError, match="S_q = 1 is the DECODE form.*kv_source='paged'"):
        GatedAttentionBlockFwd(
            mk(8, _D_MODEL),
            mk(geom.n_qkvg, _D_MODEL),
            mk(_D),
            mk(_D),
            mk(8, _ROPE),
            mk(8, _ROPE),
            mk(_D_MODEL, 32 * _D),
            mk(8, _D_MODEL),
            geom,
            thd=True,
            num_sequences=8,
            max_seq_len=1,
        )


@pytest.mark.parametrize("group", sorted(_GROUPS))
@pytest.mark.parametrize("s_q", [1, 2, 3, 4, 5])
def test_declaration_facts(group, s_q, monkeypatch):
    """A declared DECODE block records its route, its split and where the gate runs BEFORE anything compiles; the stage list
    is the serving pipeline; both projections are capture-safe; ``seq_lens_present`` reads True whatever was passed.  On the
    decode tile the gate rides the split combine (the split floored at 2); on the prefill route stage (4) is the EXTEND form's
    THD binding at the dense step's envelope, unsplit, the gate in the paged body's epilogue.  The RUBIN reading of the route on any
    host (the declaration reads the device's line since 2026-10-29: a cc 10.0 host would record the SM100 table -- its own facts row
    is ``test_on_a_cc_10_0_reading_the_declaration_records_the_sm100_route``)."""
    monkeypatch.setattr(_api, "_route_device_cc", lambda device: None)
    # Pin the split off CUDA: sm_count=0 leaves the gate outside the combine.
    blk, _ = _make_block(group, 4, s_q, fuse_gate=True, fuse_norm_rope=True, decode_split_kv=None if torch.cuda.is_available() else 2)
    r = blk.decode_route
    assert r == decode_route(blk.geom, s_q) and blk.kv_source == "paged" and blk.max_kv_len == 64 * 16 and blk.seq_lens_present is True
    assert blk._sdpa.decode_route == r and blk._sdpa.decode_split_kv == blk.decode_split_kv and blk.gate_fused and blk._gate is None and blk._sdpa.fuse_gate
    if r.decode_tile:
        assert blk._sdpa.paged_dense and blk._sdpa.seq_lens_present and not blk._sdpa.thd and blk.decode_split_kv >= 2
    else:
        assert r.prefill_tile and blk._sdpa.thd and not blk._sdpa.paged_dense and not blk._sdpa.seq_lens_present and blk.decode_split_kv == 1
        assert (blk._sdpa.num_sequences, blk._sdpa.max_seq_len, blk._sdpa.cu_seqlens) == (4, s_q, False)
    names = [st.name for st in blk._stages]
    # C0: at decode row counts (B x S_q = 4 x s_q <= 128) the declaration fuse_norm_rope=True LOWERS stage (1) to the unfused chain --
    # the slab GEMM, the norm + RoPE pass with the paged write-through folded in (fuse_cache_write_small_m's default below 128 rows: the
    # separate cache-write stage is not built) -- and reads it back; fuse_cache_write_small_m=False keeps the separate launch
    assert blk.fuse_norm_rope and not blk.fuse_norm_rope_effective and blk.small_m_decode and blk.fuse_norm_rope_small_m is None
    assert not blk.cache_write_fused and blk.cache_write_in_norm_rope and blk._cache_write is None and blk._norm_rope is not None
    assert names == ["qkv_gate_proj", "qk_norm_rope", "sdpa"] + ([] if blk.gate_fused else ["sigmoid_gate"]) + ["out_proj"], names
    sep, _ = _make_block(
        group, 4, s_q, fuse_gate=True, fuse_norm_rope=True, decode_split_kv=None if torch.cuda.is_available() else 2, fuse_cache_write_small_m=False
    )
    assert not sep.cache_write_in_norm_rope and isinstance(sep._cache_write, _CacheWrite) and sep.gate_fused is blk.gate_fused
    assert [st.name for st in sep._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa"] + ([] if sep.gate_fused else ["sigmoid_gate"]) + [
        "out_proj"
    ]
    # the A/B's control arm: fuse_norm_rope_small_m=True keeps the fused fork, the write-through FOLDED into its epilogue under AUTO
    blk_f, _ = _make_block(
        group, 4, s_q, fuse_gate=True, fuse_norm_rope=True, decode_split_kv=None if torch.cuda.is_available() else 2, fuse_norm_rope_small_m=True
    )
    assert blk_f.fuse_norm_rope_effective and blk_f.fuse_norm_rope_small_m is True and blk_f._norm_rope is None
    assert blk_f.cache_write_fused and blk_f._cache_write is None and blk_f._proj.kv_pool_page_size == blk_f.paged_kv_page_size
    assert [st.name for st in blk_f._stages] == ["qkv_gate_proj_norm_rope", "sdpa"] + ([] if blk_f.gate_fused else ["sigmoid_gate"]) + ["out_proj"]
    blk_s, _ = _make_block(
        group,
        4,
        s_q,
        fuse_gate=True,
        fuse_norm_rope=True,
        decode_split_kv=None if torch.cuda.is_available() else 2,
        fuse_cache_write=False,
        fuse_norm_rope_small_m=True,
    )
    assert not blk_s.cache_write_fused and [st.name for st in blk_s._stages] == ["qkv_gate_proj_norm_rope", "cache_write", "sdpa"] + (
        [] if blk_s.gate_fused else ["sigmoid_gate"]
    ) + ["out_proj"]
    # an explicit fuse_cache_write=True is a REQUEST only the fused fork honours (the write-through arm lives in its epilogue): under the
    # AUTO lowering the request outranks the heuristic and the fork runs -- never the separate write, never a decline
    blk_w, _ = _make_block(group, 4, s_q, fuse_gate=True, fuse_norm_rope=True, decode_split_kv=None if torch.cuda.is_available() else 2, fuse_cache_write=True)
    assert blk_w.fuse_cache_write is True and blk_w.fuse_norm_rope_small_m is None and blk_w.small_m_decode and blk_w.fuse_norm_rope_effective
    assert blk_w.cache_write_fused and blk_w._cache_write is None and blk_w._norm_rope is None and blk_w._proj.kv_pool_page_size == blk_w.paged_kv_page_size
    assert [st.name for st in blk_w._stages] == [st.name for st in blk_f._stages]
    # the explicit contradiction -- the lowering asked for by name AND the arm that lives in the fork -- is the typed decline, naming both
    with pytest.raises(NotImplementedError, match="fuse_cache_write=True with fuse_norm_rope_small_m=False"):
        _make_block(group, 4, s_q, fuse_norm_rope=True, fuse_cache_write=True, fuse_norm_rope_small_m=False)
    with pytest.raises(ValueError, match="fuse_norm_rope_small_m=True: the knob decides"):
        _make_block(group, 4, s_q, fuse_norm_rope=False, fuse_norm_rope_small_m=True)
    with pytest.raises(TypeError, match="fuse_norm_rope_small_m must be None"):
        _make_block(group, 4, s_q, fuse_norm_rope=True, fuse_norm_rope_small_m=1)
    assert blk._out_proj.capture_safe is True and (not isinstance(blk._proj, _Projection) or blk._proj.capture_safe is True)
    # the pinned split, the unfused gate, the unfused norm + RoPE
    blk1, _ = _make_block(group, 4, s_q, fuse_gate=True, decode_split_kv=1)
    if r.decode_tile:
        # an unsplit decode tile: the gate is its own launch
        assert blk1.decode_split_kv == 1 and not blk1.gate_fused and isinstance(blk1._gate, _SigmoidGate) and not blk1._sdpa.fuse_gate
        assert [st.name for st in blk1._stages] == ["qkv_gate_proj", "qk_norm_rope", "sdpa", "sigmoid_gate", "out_proj"]
        blk4, _ = _make_block(group, 4, s_q, fuse_gate=False, decode_split_kv=4)
        assert blk4.decode_split_kv == 4 and blk4._sdpa.decode_split_kv == 4 and not blk4.gate_fused
    else:
        # the prefill route runs unsplit with the gate in the body; a pinned split is the decode tile's and a typed decline here
        assert blk1.decode_split_kv == 1 and blk1.gate_fused and blk1._gate is None and blk1._sdpa.fuse_gate
        assert [st.name for st in blk1._stages] == ["qkv_gate_proj", "qk_norm_rope", "sdpa", "out_proj"]
        with pytest.raises(NotImplementedError, match="prefill route.*unsplit"):
            _make_block(group, 4, s_q, fuse_gate=False, decode_split_kv=4)


def test_the_small_m_lowering_follows_the_row_count_and_leaves_the_other_forms_alone():
    """C0's envelope: the dense DECODE form with ``fuse_norm_rope=True`` lowers stage (1) to the unfused chain exactly when
    ``B x S_q <= SMALL_M_MAX`` (the small-M route's bound), bf16 / f16, no indexer band, 16-bit pools; past the bound, and under
    ``fuse_norm_rope_small_m=True``, the fused fork runs as declared; ``fuse_norm_rope=False`` has nothing to lower (the knob is
    refused there and on a ``kv_source="self"`` block); the workspace layout does not read the fusion, so the carve is one either way."""
    from cudnn.gated_attention_block.kernels.proj_gemm import SMALL_M_MAX

    assert SMALL_M_MAX == 128
    geom, _ = _geom(12)
    at = lambda b, s_q, **kw: _make_block(12, b, s_q, fuse_norm_rope=True, **kw)[0]  # noqa: E731
    low = at(32, 4)  # 128 rows: the bound, lowered
    assert low.small_m_decode and not low.fuse_norm_rope_effective and not low.cache_write_fused and low._norm_rope is not None
    high = at(129, 1)  # 129 rows: past the bound, the fused fork as declared
    assert not high.small_m_decode and high.fuse_norm_rope_effective and high.cache_write_fused and high._norm_rope is None
    ctl = at(32, 4, fuse_norm_rope_small_m=True)
    assert ctl.small_m_decode and ctl.fuse_norm_rope_effective and ctl.cache_write_fused and ctl._norm_rope is None
    explicit = at(32, 4, fuse_norm_rope_small_m=False)
    assert explicit.fuse_norm_rope_small_m is False and not explicit.fuse_norm_rope_effective
    # an explicit write-through request keeps the fork under AUTO (user request > heuristic proposal); past the bound it changes nothing
    asked = at(32, 4, fuse_cache_write=True)
    assert asked.small_m_decode and asked.fuse_norm_rope_effective and asked.cache_write_fused and asked._norm_rope is None and asked._cache_write is None
    assert at(129, 1, fuse_cache_write=True).fuse_norm_rope_effective and at(32, 4, fuse_cache_write=False).fuse_norm_rope_effective is False
    with pytest.raises(NotImplementedError, match="fuse_cache_write=True with fuse_norm_rope_small_m=False"):
        at(32, 4, fuse_cache_write=True, fuse_norm_rope_small_m=False)
    assert at(129, 1, fuse_norm_rope_small_m=False).fuse_norm_rope_effective, "False lowers at decode row counts only; past the bound the fork stays"
    unfused = _make_block(12, 4, 4, fuse_norm_rope=False)[0]
    assert not unfused.fuse_norm_rope and not unfused.fuse_norm_rope_effective and unfused._norm_rope is not None
    assert low._layout() == ctl._layout(), "the carve does not read the fusion"
    with pytest.raises(ValueError, match="fuse_norm_rope_small_m=False: the knob decides"):
        _make_block(12, 4, 4, fuse_norm_rope=False, fuse_norm_rope_small_m=False)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    with pytest.raises(ValueError, match="fuse_norm_rope_small_m=True: the knob decides"):
        GatedAttentionBlockFwd(
            mk(1, 4, _D_MODEL),
            mk(geom.n_qkvg, _D_MODEL),
            mk(_D),
            mk(_D),
            mk(1, 4, _ROPE),
            mk(1, 4, _ROPE),
            mk(_D_MODEL, 24 * _D),
            mk(1, 4, _D_MODEL),
            geom,
            fuse_norm_rope=True,
            fuse_norm_rope_small_m=True,
        )


def test_a_self_block_keeps_its_contract_and_refuses_the_paged_read_inputs():
    """A ``kv_source="self"`` block (every block before this form): gate_fused == fuse_gate, no route, the paged-read
    declaration inputs refused rather than ignored."""
    geom, _ = _geom(16)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    args = (
        mk(2, 64, _D_MODEL),
        mk(geom.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(2, 64, _ROPE),
        mk(2, 64, _ROPE),
        mk(_D_MODEL, 32 * _D),
        mk(2, 64, _D_MODEL),
        geom,
    )
    blk = GatedAttentionBlockFwd(*args, fuse_gate=True)
    assert blk.kv_source == "self" and blk.decode_route is None and blk.gate_fused is True and blk._gate is None and not blk._sdpa.paged_dense
    blk = GatedAttentionBlockFwd(*args, fuse_gate=False)
    assert blk.gate_fused is False and isinstance(blk._gate, _SigmoidGate)
    with pytest.raises(ValueError, match="sample_block_table.*belong to the EXTEND form or the DECODE form"):
        GatedAttentionBlockFwd(*args, sample_block_table=torch.zeros(2, 4, dtype=torch.int32, device=_DEV))
    with pytest.raises(ValueError, match="decode_split_kv.*belong to the EXTEND form or the DECODE form"):
        GatedAttentionBlockFwd(*args, decode_split_kv=2)


def test_execute_refusals_on_an_uncompiled_block_are_typed_and_sync_free():
    """The declaration-vs-argument contract at execute, pinned on an UNCOMPILED block (the paged-read checks run before the
    plan check): ``seq_lens`` (a Q-side length) REFUSED under ``kv_source="paged"``; ``block_table`` +
    ``kv_lens`` REQUIRED; the forms by name; the well-formed call reaches the plan check."""
    blk, c = _make_block(16, 4, 2)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    inp = (
        mk(4, 2, _D_MODEL),
        mk(blk.geom.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(4, 2, _ROPE),
        mk(4, 2, _ROPE),
        mk(_D_MODEL, 32 * _D),
        mk(4, 2, _D_MODEL),
        torch.empty(16, dtype=torch.uint8, device=_DEV),
    )
    kl = torch.zeros(4, dtype=torch.int32, device=_DEV)
    sm = torch.zeros(8, dtype=torch.int32, device=_DEV)
    good = dict(k_cache=c["k_cache"], v_cache=c["v_cache"], block_table=c["block_table"], kv_lens=kl, slot_mapping=sm)
    with _no_host_sync():
        with pytest.raises(ValueError, match="seq_lens with kv_source='paged'.*Q-side length"):
            blk.execute(*inp, seq_lens=kl, **good)
        for missing, match in (("block_table", "needs block_table at execute"), ("kv_lens", "needs kv_lens at execute")):
            with pytest.raises(ValueError, match=match):
                blk.execute(*inp, **{k: v for k, v in good.items() if k != missing})
        with pytest.raises(ValueError, match="kv_lens must be a contiguous 1-D int32"):
            blk.execute(*inp, **{**good, "kv_lens": kl.to(torch.int64)})
        with pytest.raises(ValueError, match="kv_lens must be a contiguous 1-D int32"):
            blk.execute(*inp, **{**good, "kv_lens": torch.zeros(5, dtype=torch.int32, device=_DEV)})
        with pytest.raises(ValueError, match="kv_lens must be a contiguous 1-D int32"):
            blk.execute(*inp, **{**good, "kv_lens": torch.zeros(8, dtype=torch.int32, device=_DEV)[::2]})
        with pytest.raises(ValueError, match="block_table must be a 2-D int32"):
            blk.execute(*inp, **{**good, "block_table": c["block_table"].to(torch.int64)})
        with pytest.raises(ValueError, match="block_table must be \\[B=4"):
            blk.execute(*inp, **{**good, "block_table": torch.zeros(3, 64, dtype=torch.int32, device=_DEV)})
        with pytest.raises(ValueError, match="block_table strides must be the declared"):
            blk.execute(*inp, **{**good, "block_table": torch.zeros(4, 128, dtype=torch.int32, device=_DEV)[:, :64]})
        with pytest.raises(ValueError, match="must be \\[B=4, >= max_pages=64\\]"):  # the declared row stride, too few pages
            blk.execute(*inp, **{**good, "block_table": torch.zeros(4, 64, dtype=torch.int32, device=_DEV)[:, :8]})
        with pytest.raises(ValueError, match="k_cache must match sample_k_cache exactly"):
            blk.execute(*inp, **{**good, "k_cache": torch.empty(64, 16, 2, _D, dtype=torch.bfloat16, device=_DEV).permute(0, 2, 1, 3)})
        with pytest.raises(ValueError, match="v_cache must match sample_v_cache exactly"):
            blk.execute(*inp, **{**good, "v_cache": torch.empty(32, 2, 16, _D, dtype=torch.bfloat16, device=_DEV)})
        with pytest.raises(RuntimeError, match="compile"):
            blk.execute(*inp, **good)


def test_the_stage_refuses_the_paged_inputs_on_a_self_declaration_and_the_q_side_length_on_a_paged_one(monkeypatch):
    _rubin_reading(monkeypatch)
    from cudnn.api_base import TensorDesc

    geom, _ = _geom(16)
    dev = torch.device(_DEV)
    with pytest.raises(ValueError, match="belong to kv_source='paged'"):
        _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=False, paged_page_size=16)
    with pytest.raises(ValueError, match="kv_source must be one of"):
        _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=False, kv_source="pooled")
    with pytest.raises(ValueError, match="needs paged_page_size > 0"):
        _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=False, kv_source="paged")
    kc, vc = _pools(4, 2, 16, _D, "hnd", torch.bfloat16)
    kd, vd = (TensorDesc(dtype=t.dtype, shape=tuple(t.shape), stride=tuple(t.stride()), stride_order=(3, 2, 1, 0), device=t.device) for t in (kc, vc))
    bd = TensorDesc(dtype=torch.int32, shape=(1, 4), stride=(4, 1), stride_order=(1, 0), device=dev)
    paged = dict(kv_source="paged", paged_page_size=16, k_cache_desc=kd, v_cache_desc=vd, block_table_desc=bd)
    with pytest.raises(NotImplementedError, match=f"serves S_q <= {DECODE_MAX_S_Q}.*EXTEND form"):
        _Sdpa(geom, batch=1, seq_len=33, dtype=torch.bfloat16, device=dev, want_lse=False, **paged)
    with pytest.raises(NotImplementedError, match="split combine only"):
        _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=False, fuse_gate=True, decode_split_kv=1, **paged)
    with pytest.raises(NotImplementedError, match="writes no LSE"):
        _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=True, **paged)
    st = _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=False, decode_split_kv=2, fuse_gate=True, **paged)
    assert st.paged_dense and st.decode_route.decode_tile and st.seq_lens_present and st.decode_split_kv == 2


def test_the_workspace_carve_is_unchanged_by_the_paged_read(monkeypatch):
    _rubin_reading(monkeypatch)
    """The read changes what stage (4) reads, not the carve: the SDPA's split scratch sits in the engine region (the max with
    the GEMMs'), so the intermediates' layout equals a ``kv_source="self"`` block's at the same shape."""
    blk, _ = _make_block(16, 4, 2, fuse_gate=True, fuse_norm_rope=True)
    geom, _ = _geom(16)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    ref = GatedAttentionBlockFwd(
        mk(4, 2, _D_MODEL),
        mk(geom.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(4, 2, _ROPE),
        mk(4, 2, _ROPE),
        mk(_D_MODEL, 32 * _D),
        mk(4, 2, _D_MODEL),
        geom,
        fuse_gate=True,
        fuse_norm_rope=True,
    )
    assert blk._layout() == ref._layout()


# ---------------------------------------------------------------------------
# Rubin: the cells
# ---------------------------------------------------------------------------

_SPLIT = 128  # the SDPA's KV tile


def _rope_tables(n_pos, dtype, geom_ref):
    cos, sin = build_rope_tables(n_pos, _ROPE, base=geom_ref.rope_base, batch=1, device=_DEV, dtype=dtype)
    return cos, sin  # [1, n_pos, ROPE]


class _Case:
    """One DECODE step over a batch of sequences with lengths ``kv_lens`` (0 = a padded entry): the inputs of the whole
    sequence per entry, the step's inputs (the last ``s_q`` rows), the sentinel-filled pools pre-filled with the ORACLE's
    prefix K / V, the block table (block 0 = the NULL block, never allocated), the slot mapping of the step's tokens (-1 on a
    padded entry), and the per-sequence fp32 oracle of the step's ``out`` rows."""

    def __init__(
        self,
        group,
        s_q,
        kv_lens,
        *,
        page=16,
        layout="hnd",
        dtype=torch.bfloat16,
        seed=0,
        n_pages=None,
        table_row_stride=None,
        blk_kw=None,
        heads=None,
        d_model=None,
        quant=None,
        check_numerics=True,
    ):
        """``quant`` (appended; None = the bf16 / f16 form) selects the FP8-PROJECTION + BF16-ATTENTION mode (``QuantSpec(attention_dtype=
        torch.bfloat16)``): ``True`` calibrates the static per-tensor scales on this case's own data (amax scales for h / W_qkvg / W_o, ``scale_o``
        from the oracle's gated O: an offline calibration stand-in), a ``scales`` dict of another case reuses ITS scales (a sibling case whose e4m3
        codes a captured block must read under the same spec); ``h`` and the weights are then e4m3 codes, ``dtype`` stays the bf16 activation
        dtype, and the oracle is the fp8 reference's ``attention_bf16`` arm (no Q / K / V quantization point, the gated O quantized at ``scale_o``).  ``check_numerics=False`` (appended) builds the case WITHOUT a reference -- random bf16 K / V rows fill
        the pools, ``scale_o`` is 1.0, ``ref_out`` stays 0 and ``check_against_oracle`` / ``recompute_oracle`` refuse -- for the cells that pin
        launches, not values (the census)."""
        self.group, self.s_q, self.kv_lens, self.page, self.layout, self.dtype = group, int(s_q), [int(x) for x in kv_lens], page, layout, dtype
        self.check_numerics = bool(check_numerics)
        self.quant_mode = quant is not None
        self.d_model = _D_MODEL if d_model is None else int(d_model)  # the shrunk matrix width, or a shard's real one
        self.geom, self.ref_geom = _geom(group, heads=heads, d_model=self.d_model)
        h_q, h_kv = _GROUPS[group] if heads is None else heads
        g = torch.Generator(device=_DEV).manual_seed(seed)
        B = len(self.kv_lens)
        assert all(L == 0 or L >= s_q for L in self.kv_lens), "a live entry holds at least the step's tokens"
        L_max = max(max(self.kv_lens), s_q)

        def randn(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        self.w = dict(
            w_qkvg=randn(self.geom.n_qkvg, self.d_model),
            w_q_norm=torch.ones(_D, dtype=dtype, device=_DEV),
            w_k_norm=torch.ones(_D, dtype=dtype, device=_DEV),
            w_o=randn(self.d_model, h_q * _D),
        )
        cos_t, sin_t = _rope_tables(L_max, dtype, self.ref_geom)
        self._rope = (cos_t, sin_t)
        self.h_full = [randn(1, max(L, s_q), self.d_model, std=1.0) for L in self.kv_lens]
        self.scales = self.quant = None
        if self.quant_mode:
            # the mode: e4m3 CODES for h / W_qkvg / W_o under static per-tensor scales (the same codes a serving stack's FP8 W8A8 projections hold)
            assert dtype == torch.bfloat16, "the FP8-projection + bf16-attention mode's activations are bf16"
            if quant is True:
                self.scales = dict(
                    s_h=amax_scale(torch.cat([hf.flatten() for hf in self.h_full])),
                    s_wq=amax_scale(self.w["w_qkvg"]),
                    s_wo=amax_scale(self.w["w_o"]),
                    scale_o=None,
                )
            else:
                self.scales = dict(quant)
            self.w["w_qkvg"] = quant_e4m3(self.w["w_qkvg"], self.scales["s_wq"])
            self.w["w_o"] = quant_e4m3(self.w["w_o"], self.scales["s_wo"])
            self.h_full = [quant_e4m3(hf, self.scales["s_h"]) for hf in self.h_full]
        # the step's inputs: the LAST s_q rows (and their positions' RoPE rows); a padded entry takes its row 0 .. s_q-1 (never read)
        self.h = torch.stack([hf[0, -s_q:] for hf in self.h_full])  # [B, s_q, d_model]
        pos = [list(range(L - s_q, L)) if L else list(range(s_q)) for L in self.kv_lens]
        self.cos = torch.stack([cos_t[0, p] for p in pos])
        self.sin = torch.stack([sin_t[0, p] for p in pos])
        # the oracle per live entry over the FULL sequence: its K / V rows (bf16 -- the oracle's single rounding) fill the pools
        self.ref_out = torch.zeros(B, s_q, self.d_model, dtype=dtype, device=_DEV)
        self.ref_rows = {}
        self.ref_o, self.ref_gate, self.ref_og, self.ref_o8 = {}, {}, {}, {}  # the mode's oracle intermediates (the step's rows), per live entry
        if self.quant_mode:
            if self.scales["scale_o"] is None:
                if self.check_numerics:
                    # scale_o from the oracle's gated O over every live entry (pass 1 at scale 1), then the oracle at that scale (pass 2)
                    og = torch.cat([self._mode_oracle(b, 1.0).o_gated.flatten() for b, L in enumerate(self.kv_lens) if L])
                    self.scales["scale_o"] = amax_scale(og)
                else:
                    self.scales["scale_o"] = 1.0  # no reference: any finite positive scale declares the same chain
            sc = self.scales
            self.quant = QuantSpec.bf16_attention(1.0 / sc["s_h"], 1.0 / sc["s_wq"], 1.0 / sc["s_wo"], sc["scale_o"])
            self.quant.validate()
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            if not self.check_numerics:
                # no reference (the launch census): random bf16 K / V rows fill the pools -- the kernel chain does not depend on the values
                self.ref_rows[b] = (randn(L, h_kv, _D, std=1.0), randn(L, h_kv, _D, std=1.0))
                continue
            if self.quant_mode:
                ref = self._mode_oracle(b, self.scales["scale_o"])
                self.ref_o[b], self.ref_gate[b], self.ref_og[b], self.ref_o8[b] = ref.o[0, -s_q:], ref.gate[0, -s_q:], ref.o_gated[0, -s_q:], ref.o8[0, -s_q:]
            else:
                ref = gated_attention_block_reference(
                    self.h_full[b], self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], self.w["w_o"], self.ref_geom
                )
            self.ref_out[b] = ref.out[0, -s_q:]
            self.ref_rows[b] = (ref.k[0], ref.v[0])  # [L, h_kv, D]
        # pages: block 0 reserved (NULL); each live entry gets ceil(L / P) pages from a permutation of 1 .. n_pages-1
        need = sum(-(-L // page) for L in self.kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=g, device=_DEV) + 1).tolist()
        max_pages = max(-(-L_max // page), 1)
        table_cols = max_pages if table_row_stride is None else table_row_stride
        assert table_cols >= max_pages
        big = torch.zeros(B, table_cols, dtype=torch.int32, device=_DEV)
        self.block_table = big[:, :max_pages]
        self.max_kv_len = max_pages * page
        it = iter(perm)
        slots = torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _pools(self.n_pages, h_kv, page, _D, layout, dtype)
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue  # the padded entry: row 0 (the NULL block), slots -1
            pages = [next(it) for _ in range(-(-L // page))]
            self.block_table[b, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=_DEV)
            k_rows, v_rows = self.ref_rows[b]
            for p in range(L - s_q):  # the PREFIX: the oracle's rows into the pools
                pg, off = pages[p // page], p % page
                self.k_cache[pg, :, off] = k_rows[p]
                self.v_cache[pg, :, off] = v_rows[p]
            for i in range(s_q):  # the step's tokens: written through by the block
                p = L - s_q + i
                slots[b * s_q + i] = pages[p // page] * page + p % page
        self.slot_mapping = slots
        self.kv_lens_t = torch.tensor(self.kv_lens, dtype=torch.int32, device=_DEV)
        self.block0_bits = _bits(self.k_cache[0]).clone(), _bits(self.v_cache[0]).clone()
        self.live = torch.tensor([L > 0 for L in self.kv_lens], device=_DEV)
        # the fused norm + RoPE fork is rendered for bf16 only: f16 cells run the unfused (1) then (2)+(3)
        kw = dict(
            paged_kv_page_size=page,
            kv_source="paged",
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            fuse_gate=True,
            # the mode runs the UNFUSED stage (1) (fuse_norm_rope is refused there: no bf16 slab under the fork)
            fuse_norm_rope=dtype == torch.bfloat16 and not self.quant_mode,
            **({"quant": self.quant} if self.quant_mode else {}),
        )
        kw.update(blk_kw or {})
        self.blk_kw = kw
        self.out = torch.empty(B, s_q, self.d_model, dtype=dtype, device=_DEV)
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, self.geom, **kw
        )
        self.ws = None

    def _mode_oracle(self, b, scale_o):
        """The mode's oracle over live entry ``b``'s FULL sequence: the fp8 reference's ``attention_bf16`` arm on this case's e4m3 codes
        and scales, the gated O quantized at ``scale_o`` (``Fp8RefOutputs``: out, the bf16 K / V rows, the pre-gate O, the gate, the gated O, o8)."""
        L, sc = self.kv_lens[b], self.scales
        cos_t, sin_t = self._rope
        inp = dict(
            h=self.h_full[b],
            w_qkvg=self.w["w_qkvg"],
            w_o=self.w["w_o"],
            w_q_norm=self.w["w_q_norm"],
            w_k_norm=self.w["w_k_norm"],
            cos=cos_t[:, :L],
            sin=sin_t[:, :L],
        )
        return gated_attention_block_fp8_reference(
            inp,
            self.ref_geom,
            descale_h=1.0 / sc["s_h"],
            descale_w_qkvg=1.0 / sc["s_wq"],
            descale_w_o=1.0 / sc["s_wo"],
            scale_q=None,
            scale_k=None,
            scale_v=None,
            scale_o=scale_o,
            attention_bf16=True,
            return_rows=True,
        )

    def recompute_oracle(self, scale_o):
        """The mode's ``out`` oracle at another ``scale_o`` (the recalibration cells), ``[B, s_q, d_model]`` bf16 with the dead rows 0."""
        assert self.check_numerics, "a check_numerics=False case has no reference (the launch census)"
        out = torch.zeros_like(self.ref_out)
        for b, L in enumerate(self.kv_lens):
            if L:
                out[b] = self._mode_oracle(b, scale_o).out[0, -self.s_q :]
        return out

    def compile(self):
        self.blk.check_support()
        self.blk.compile()
        self.ws = torch.empty(self.blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
        return self

    def execute(self, *, out=None, ws=None, h=None, cos=None, sin=None, kv_lens=None, block_table=None, slot_mapping=None, stream=None):
        out = self.out if out is None else out
        self.blk.execute(
            self.h if h is None else h,
            self.w["w_qkvg"],
            self.w["w_q_norm"],
            self.w["w_k_norm"],
            self.cos if cos is None else cos,
            self.sin if sin is None else sin,
            self.w["w_o"],
            out,
            self.ws if ws is None else ws,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.block_table if block_table is None else block_table,
            kv_lens=self.kv_lens_t if kv_lens is None else kv_lens,
            slot_mapping=self.slot_mapping if slot_mapping is None else slot_mapping,
            current_stream=stream,
        )
        return out

    def run(self):
        if self.ws is None:
            self.compile()
        self.execute()
        torch.cuda.synchronize()
        return self.out

    # -- views of the block's intermediates after an execute --------------------------------------------------------------
    def slab_views(self, ws=None):
        """The post-norm-RoPE Q band and the GATE band of the slab the last execute left in the workspace, as the SDPA's
        ``[B, S_q, H_q, D]`` views (the slab's token stride), and the O slot."""
        ws = self.ws if ws is None else ws
        g, lay = self.geom, self.blk._ws
        t = self.blk.batch * self.blk.seq_len
        proj = _view(ws, lay.proj, (t, g.n_qkvg), self.dtype)
        o_q, o_g = g.qkvg_offsets[:2]
        q = _cols(proj, o_q, g.h_q, g.d_head).view(self.blk.batch, self.blk.seq_len, g.h_q, g.d_head)
        gate = _cols(proj, o_g, g.h_q, g.d_head).view(self.blk.batch, self.blk.seq_len, g.h_q, g.d_head)
        o = _view(ws, lay.o, (t, g.h_q, g.d_head), self.dtype).view(self.blk.batch, self.blk.seq_len, g.h_q, g.d_head)
        return q, gate, o, ws[lay.engine_scratch :]

    def o8_view(self, ws=None):
        """The mode's e4m3 gated O the last execute left in the workspace's ``o8`` slot, as ``[B, S_q, H_q, D]`` (the FP8 out projection's A operand)."""
        ws = self.ws if ws is None else ws
        g, lay = self.geom, self.blk._ws
        assert lay.o8 >= 0, "no o8 slot: not a quantized declaration"
        t = self.blk.batch * self.blk.seq_len
        return _view(ws, lay.o8, (t, g.h_q, g.d_head), torch.float8_e4m3fn).view(self.blk.batch, self.blk.seq_len, g.h_q, g.d_head)

    def check_against_oracle(self, out=None, *, what="out"):
        assert self.check_numerics, "a check_numerics=False case has no reference (the launch census)"
        out = self.out if out is None else out
        assert torch.isfinite(out.float()).all(), f"{what}: non-finite"
        dead = ~self.live
        if dead.any():
            assert (out[dead] == 0).all(), f"{what}: a padded entry (kv_lens = 0) must yield out rows EXACTLY 0"
        c = _cos(out[self.live], self.ref_out[self.live])
        md = (out[self.live].float() - self.ref_out[self.live].float()).abs().max().item()
        print(f"[{self.group}/{self.s_q} p{self.page} {self.layout} {self.dtype} lens={self.kv_lens}] {what} vs oracle cos {c:.6f} max|diff| {md:.3e}")
        assert c > 0.999, f"{what}: cos {c}"
        return c, md

    def route_readback(self):
        """What the adapter COMPILED, read back: the kernel template, the module's packing / split / gate facts."""
        impl = self.blk._sdpa._impl
        mod = impl._k_mod
        return dict(
            template=impl.kernel_template,
            file=mod.__file__,
            heads_per_tile=getattr(mod, "HEADS_PER_TILE", None),
            split=getattr(getattr(mod, "CFG", None), "SPLIT_KV", None),
            epilogue_gate=getattr(getattr(mod, "PARAMS", None), "epilogue_gate", None),
            n_q=getattr(getattr(mod, "CFG", None), "N_Q", None),
            adapter_split=impl.split_kv,
            adapter_pack=impl.pack_gqa,
        )


_CASES = {}


def _case(key, *args, **kw):
    if key not in _CASES:
        _CASES[key] = _Case(*args, **kw).compile()
    return _CASES[key]


# kv_lens of the standard cell: one KV tile (100: a tail tile too), ring depth 3 (384 = 3 tiles exactly), ring depth 4 with a
# tail (500), a PADDED entry (0: block 0 + slot -1) -- kv_lens > S_q on every live row at S_q <= 5.
_LENS = (100, 384, 500, 0)


def _std_case(group, s_q, **kw):
    return _case(("std", group, s_q) + tuple(sorted(kw.items())), group, s_q, _LENS, **kw)


@requires_rubin
@pytest.mark.parametrize("group", sorted(_GROUPS))
@pytest.mark.parametrize("s_q", [1, 2, 3, 4, 5])
def test_decode_matches_the_oracle_at_every_group_and_s_q(group, s_q):
    """The DECODE cells of the served-group matrix: every served group x S_q 1 .. 4 (the claimed envelope on the decode tile; at 32 / 2
    S_q 3 / 4 ride the packed PREFILL tile under the default policy) and S_q = 5 (the packed prefill tile at 16 / 8, the adapter's
    UNPACKED tile at 12 -- one head per CTA --, the 32-column tile at 6 / 4 -- correct and DECLARED), against the per-sequence fp32
    oracle under the dense suite's budget (cos > 0.999 on ``out``), the padded entry EXACTLY 0 with block 0 untouched, the route
    pinned against the compiled kernel (its template, packing, split and where the gate runs: the combine on the decode tile, the
    paged body's epilogue on the prefill tile)."""
    case = _std_case(group, s_q)
    case.run()
    case.check_against_oracle()
    r, rb = case.blk.decode_route, case.route_readback()
    assert rb["template"] == r.tile, rb
    assert rb["adapter_split"] == case.blk.decode_split_kv and (rb["adapter_pack"] is True) == r.pack_gqa, (rb, r)
    assert (r.tile, r.n_q, r.token_units, r.pack_gqa) == _ROUTE_TABLE[group][s_q], (r, _ROUTE_TABLE[group][s_q])
    if r.decode_tile:
        assert rb["file"].endswith("/sm107/decode_d256_f16.py") and rb["heads_per_tile"] == (group if r.pack_gqa else 1), rb
        assert rb["n_q"] == r.n_q and rb["split"] == case.blk.decode_split_kv, rb
        assert rb["epilogue_gate"] is False and case.blk.gate_fused and case.blk.decode_split_kv >= 2, "the gate rides the COMBINE, never the tile"
    else:
        assert r.prefill_tile and rb["file"].endswith("/sm100/prefill_d256_f16.py") and rb["heads_per_tile"] == group and rb["n_q"] is None, rb
        assert rb["epilogue_gate"] is True and case.blk.gate_fused and case.blk.decode_split_kv == 1, "the gate rides the paged body's epilogue"
    assert r.claimed is (
        s_q <= DECODE_CLAIMED_MAX_S_Q and r.pack_gqa and r.decode_tile and r.n_q <= DECODE_CLAIMED_MAX_N_Q
    )  # the 64-column cells: routed, not claimed
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0), "block 0 (the NULL block) must never be written"


@requires_rubin
@pytest.mark.parametrize("s_q", [1, 4], ids=["sq1", "sq4"])
def test_the_fused_fork_control_arm_matches_the_oracle_at_decode_row_counts(s_q):
    """``fuse_norm_rope_small_m=True`` (the A/B's control arm) runs the fused norm + RoPE projection with the write-through folded in at
    decode row counts, where the default declaration lowers to the unfused chain: both within the oracle's budget (two kernels for
    stage (1), two roundings -- not bitwise), the pools' step rows and the padded entry exactly as on the default chain, the stage
    lists the two chains'."""
    ctl = _case(("ctl", 12, s_q), 12, s_q, _LENS, blk_kw=dict(fuse_norm_rope_small_m=True))
    dflt = _std_case(12, s_q)
    assert ctl.blk.fuse_norm_rope_effective and ctl.blk.cache_write_fused and not dflt.blk.fuse_norm_rope_effective and not dflt.blk.cache_write_fused
    assert [st.name for st in ctl.blk._stages] == ["qkv_gate_proj_norm_rope", "sdpa", "out_proj"]
    assert [st.name for st in dflt.blk._stages] == ["qkv_gate_proj", "qk_norm_rope", "sdpa", "out_proj"]  # the write-through in the norm + RoPE launch
    a = ctl.run().clone()
    b = dflt.run().clone()
    ctl.check_against_oracle(a, what="the fused fork (control)")
    dflt.check_against_oracle(b, what="the lowered chain (default)")
    assert torch.equal(a[~ctl.live], b[~dflt.live]) and (a[~ctl.live] == 0).all()
    assert _cos(a[ctl.live], b[dflt.live]) > 0.9999
    # the explicit write-through request (fuse_cache_write=True under AUTO) runs the same fork: the same stage list, BITWISE the control arm
    asked = _case(("asked", 12, s_q), 12, s_q, _LENS, blk_kw=dict(fuse_cache_write=True))
    assert asked.blk.fuse_norm_rope_small_m is None and asked.blk.fuse_norm_rope_effective and asked.blk.cache_write_fused
    assert [st.name for st in asked.blk._stages] == [st.name for st in ctl.blk._stages]
    c = asked.run().clone()
    asked.check_against_oracle(c, what="the fused fork (the explicit write-through request)")
    assert torch.equal(c, a), "the explicit request and the control arm run the same kernels: bitwise"
    assert torch.equal(_bits(asked.k_cache), _bits(ctl.k_cache)) and torch.equal(_bits(asked.v_cache), _bits(ctl.v_cache))


@requires_rubin
def test_s_q_five_rides_the_prefill_tile_at_sixteen_the_64_column_tile_at_twelve_and_eight_and_the_32_column_tile_at_six_and_four():
    """S_q = 5 at every served group: past the single-unit table (32 / 2 = 80 rows) the step rides TWO units of the 64-column tile at
    the standard cell's four sequences (8 CTAs: below the prefill route's fill) and the paged PREFILL tile with the whole group packed
    (HEADS_PER_TILE = the group) at 32 sequences (64 CTAs: the fill); at 24 / 2 (60 rows) and 8 / 1 (40 rows) ONE unit of the
    64-column tile (the width that replaced the unpacked 12-stream arm at 24 / 2); at 24 / 4 (30 rows) and 4 / 1 (20 rows) one unit
    of the 32-column tile.  Every arm declared by the route and pinned on the compiled module (the width too), the route's batch read back."""
    for group in sorted(_GROUPS):
        case = _std_case(group, 5)
        r, rb = case.blk.decode_route, case.route_readback()
        assert r.tile == rb["template"] == DECODE_TILE and r.pack_gqa and r.batch == 4 and r.token_units == (2 if group == 16 else 1) == r.kv_streams
        assert rb["heads_per_tile"] == group and rb["adapter_pack"] is True, (group, r, rb)
        assert r.n_q == {16: 64, 12: 64, 8: 64, 6: 32, 4: 32}[group] and rb["n_q"] == r.n_q, (group, r, rb)
    fill = _case(("fill", 16, 5), 16, 5, _LENS * 8)  # 32 sequences x 2 KV heads = 64 CTAs: the prefill route's fill
    fill.run()
    fill.check_against_oracle(what="prefill route at the fill")
    r, rb = fill.blk.decode_route, fill.route_readback()
    assert r.prefill_tile and r.tile == rb["template"] and r.batch == 32 and r.kv_streams == 1 and r.token_units == 0 and r.n_q == 0, (r, rb)
    assert rb["heads_per_tile"] == 16 and rb["adapter_pack"] is True and rb["file"].endswith("sm100/prefill_d256_f16.py"), rb


def test_past_s_q_32_the_step_is_a_typed_decline_the_envelope_predicts():
    """A dense step past the SERVED envelope (``S_q = 33``: past the unpacked arm's 32-column tile) is the EXTEND form's packed business
    over the pools: the route says not served with the reason (the EXTEND form), the declaration raises it, the envelope predicts it --
    no silent fall-through; ``S_q = 32`` is served at every group (the decode tile, unpacked or packed, or the prefill tile)."""
    for group in sorted(_GROUPS):
        geom, _ = _geom(group)
        r = decode_route(geom, DECODE_MAX_S_Q + 1)
        assert not r.served and r.tile is None and "EXTEND form" in r.reason and r.n_q == 0 and r.kv_streams == 0
        assert decode_route(geom, DECODE_MAX_S_Q).served and decode_route(geom, DECODE_MAX_S_Q).s_q <= DECODE_MAX_ROWS
        with pytest.raises(NotImplementedError, match=f"serves S_q <= {DECODE_MAX_S_Q}.*EXTEND form"):
            _make_block(group, 2, DECODE_MAX_S_Q + 1)
        env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
        assert not env.decode.route(DECODE_MAX_S_Q + 1).served and env.decode.max_rows == DECODE_MAX_ROWS and env.decode.max_s_q == DECODE_MAX_S_Q
        assert env.decode.route(DECODE_MAX_S_Q).served and env.decode.route(9).served


# The 397B shards at their REAL width (d_model 4096; the layout contract pins n_qkvg 8704 / 4608 / 2560): 16 / 1 (TP 2) rides group
# 16's route with ONE KV head, 8 / 1 (TP 4, the serving deployment) group 8's, 4 / 1 (TP 8) group 4's -- one kernel block size per
# shard, so the three cells also walk the page sizes the deployments allocate.  (h_q, h_kv, page)
_SHARDS = {"q35_397b_tp2_16_1": (16, 1, 16), "q35_397b_tp4_8_1": (8, 1, 32), "q35_397b_tp8_4_1": (4, 1, 64)}
_D_MODEL_397B = 4096


def test_the_397b_shards_ride_their_groups_routes_and_the_envelope_serves_both_forms(monkeypatch):
    """A shard's decode route is its GROUP's (the rule reads ``h_q // h_kv`` and ``d_head`` only, never ``d_model`` or the KV head
    count): 16 / 1 = the 32 / 2 rows, 8 / 1 = group 8's, 4 / 1 = group 4's (the 16- then the 32-column tile in one unit up to
    ``S_q = 8``, a typed decline past that); the envelope on a Rubin reading serves the EXTEND and the DECODE form at every shard."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    for name, (h_q, h_kv, _page) in _SHARDS.items():
        group = h_q // h_kv
        geom, _ = _geom(group, heads=(h_q, h_kv), d_model=_D_MODEL_397B)
        small, _ = _geom(group)
        assert geom.n_qkvg % 256 == 0 and geom.d_model == _D_MODEL_397B
        for s_q in list(range(1, 11)) + [16, 17, 32, 33]:
            r, t = decode_route(geom, s_q), decode_route(small, s_q)
            assert (r.group, r.n_q, r.token_units, r.pack_gqa, r.claimed, r.served) == (group, t.n_q, t.token_units, t.pack_gqa, t.claimed, t.served), (
                name,
                s_q,
                r,
                t,
            )
        for s_q, (tile, n_q, units, packed) in _ROUTE_TABLE[group].items():
            r = decode_route(geom, s_q)
            assert (r.tile, r.n_q, r.token_units, r.pack_gqa) == (tile, n_q, units, packed), (name, s_q, r)
        env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
        assert env.forms == ("extend", "decode") and env.extend.served and env.decode.served, (name, env.extend.reason, env.decode.reason)
        assert env.decode.route(1).claimed and env.decode.route(DECODE_MAX_S_Q).served and not env.decode.route(DECODE_MAX_S_Q + 1).served
        fill = -(-DECODE_PREFILL_ROUTE_MIN_CTAS // h_kv)
        assert all(
            env.decode.route(s, batch=fill).kv_streams == 1 for s in range(1, 9)
        ), "every shard's group divides the prefill tile: one KV stream at every S_q <= 8 once the step fills the part"
        assert all(env.decode.route(s).kv_streams <= 2 for s in range(1, 9)), "below the fill at most the two-unit arm's streams"


@requires_rubin
@pytest.mark.parametrize("name", list(_SHARDS), ids=list(_SHARDS))
def test_qwen35_shards_decode_match_the_oracle(name):
    """One DECODE step (``S_q = 1``: the pure decode step, inside the claimed envelope) per 397B shard at d_model 4096 -- 16 / 1 at
    page 16, 8 / 1 at page 32, 4 / 1 at page 64 -- over the standard lengths (a tail tile, the ring depth, four tiles with a tail, a
    padded entry) against the per-sequence fp32 oracle under the dense suite's budget; the route pinned on the compiled kernel (the
    16-column tile with the WHOLE group packed per tile: HEADS_PER_TILE 16 / 8 / 4 over the one KV head), the gate in the combine,
    block 0 untouched."""
    h_q, h_kv, page = _SHARDS[name]
    group = h_q // h_kv
    case = _case(("shard", name), group, 1, _LENS, page=page, heads=(h_q, h_kv), d_model=_D_MODEL_397B)
    case.run()
    case.check_against_oracle(what=f"{name} p{page} out")
    r, rb = case.blk.decode_route, case.route_readback()
    assert r.claimed and r.pack_gqa and r.n_q == 16 and r.token_units == 1, r
    assert rb["template"] == r.tile == DECODE_TILE and rb["file"].endswith("/sm107/decode_d256_f16.py"), rb
    assert rb["heads_per_tile"] == group and rb["adapter_pack"] is True and rb["n_q"] == 16, rb
    assert rb["epilogue_gate"] is False and case.blk.gate_fused and case.blk.decode_split_kv >= 2, "the gate rides the COMBINE, never the tile"
    assert case.geom.d_model == _D_MODEL_397B and case.geom.h_kv == 1 and case.k_cache.shape[1] == 1
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0), "block 0 (the NULL block) must never be written"


@requires_rubin
@pytest.mark.parametrize("lens", [(128, 384, 512, 1024), (2, 300, 130, 256)], ids=["n_kv_1_3_4_8", "prefix_empty_and_tails"])
def test_ring_wrap_and_tail_lengths(lens):
    """n_kv = 1 / 3 (the ring depth) / 4 / 8 exactly, then an entry whose WHOLE content is the step (kv_lens == S_q: an empty
    prefix), lengths off the tile and the page."""
    case = _case(("lens", lens), 16, 2, lens)
    case.run()
    case.check_against_oracle()


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("page", SERVING_PAGE_SIZES)
def test_l1_l2_l3_pools_are_bitwise_on_the_decode_tile_at_every_page_size(page, dtype):
    """The pool-layout matrix on the decode tile (the tracker's matrix): the three stride patterns holding the same values give a BITWISE ``out``
    at pages 16 / 32 / 64 / 128, bf16 and f16 (f16 on the unfused norm + RoPE), and each matches the oracle."""
    outs = {}
    for layout in _LAYOUTS:
        case = _case(("layout", page, layout, dtype), 12, 2, (200, 0, 77, 1000), page=page, layout=layout, dtype=dtype, seed=3)
        outs[layout] = case.run().clone()
        case.check_against_oracle(what=f"{layout} p{page} {dtype}")
    for layout in _LAYOUTS[1:]:
        assert torch.equal(
            outs[layout], outs["hnd"]
        ), f"{layout} != hnd at page {page} {dtype}: max|diff| {(outs[layout].float() - outs['hnd'].float()).abs().max().item():.3e}"


@requires_rubin
def test_padded_entries_yield_exact_zero_and_write_nothing_to_block_zero():
    """The serving stacks' padded full-graph entry: ``kv_lens = 0`` + block table row 0 + slot -1 at EVERY padded
    position of a batch (first, middle, last), ``out`` rows exactly 0, block 0's bytes untouched, nothing written anywhere
    for the padded tokens (the live entries' rows are the only finite ones)."""
    lens = (0, 300, 0, 129, 0)
    case = _case(("padded", lens), 8, 2, lens, seed=7)
    before = _written(case.k_cache)
    case.run()
    case.check_against_oracle()
    assert (case.out[~case.live] == 0).all()
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0)
    # the write-through landed exactly the live entries' step tokens (S_q rows x H_kv heads each) and nothing else
    assert _written(case.k_cache) == before + sum(2 for L in lens if L) * _GROUPS[8][1]


def _written(pool):
    """How many ``[D]`` rows of a sentinel-filled pool were written (a written row is no longer the sentinel end to end)."""
    return int((pool.float() != _SENTINEL).all(-1).sum().item())


@requires_rubin
def test_a_block_table_row_slice_at_an_arbitrary_row_stride_binds_as_a_view():
    """A block-table row slice: the serving stack's table is ``[num_reqs, max_blocks_per_req]`` and the decode batch's rows are a SLICE of it;
    the block declares and binds that view (its row stride compiled in) -- bitwise the compact table."""
    lens = (300, 0, 500, 77)  # 32 pages at most: the 37-wide table rows leave 5 unused columns past the declared max_pages
    compact = _case(("rowstride", "compact"), 16, 3, lens, seed=5)
    sliced = _case(("rowstride", "sliced"), 16, 3, lens, seed=5, table_row_stride=37)
    assert (
        sliced.block_table.stride(0) == 37
        and tuple(sliced.blk._descs["block_table"].stride) == (37, 1)
        and compact.block_table.stride(0) == compact.block_table.shape[1]
    )
    a, b = compact.run().clone(), sliced.run().clone()
    compact.check_against_oracle()
    assert torch.equal(a, b)


def _adapter_like(case, *, sample_k, sample_v, paged, gate, split, lse=False):
    """A standalone ``SdpaFwdDslSm100`` declared like the block's stage (4): the SAME Q view strides, the same packing, split,
    diagonal, scale and scheduler policy; ``paged`` = the pools + table, else a padded dense cache."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    blk = case.blk
    g = case.geom
    q, gate_v, o, _ = case.slab_views()
    o_buf = torch.empty_like(o)
    kw = dict(
        sample_q=q.transpose(1, 2),
        sample_k=sample_k,
        sample_v=sample_v,
        sample_o=o_buf.transpose(1, 2),
        sample_lse=torch.empty(blk.batch, g.h_q, blk.seq_len, dtype=torch.float32, device=_DEV) if lse else None,
        is_causal=g.is_causal,
        causal_bottom_right=g.causal_bottom_right,
        scale_softmax=g.scale,
        seq_kv_lens_present=True,
        sched_policy=blk._sdpa._impl.sched_policy,
        split_kv=split,
        pack_gqa=True if blk.decode_route.pack_gqa else None,
    )
    if paged:
        bts = tuple(int(x) for x in blk._descs["block_table"].stride)
        kw.update(paged_page_size=case.page, paged_max_seq_len_kv=case.max_kv_len, paged_table_stride=bts, paged_table_v_stride=bts)
    if gate:
        kw["sample_gate"] = gate_v.transpose(1, 2)
    api = SdpaFwdDslSm100(**kw)
    api.check_support()
    api.compile()
    return api, o_buf, q, gate_v


@requires_rubin
def test_the_block_form_equals_the_standalone_adapter_bitwise_on_the_same_views():
    """The block's DECODE form vs the standalone adapter on the SAME Q views (the slab band at its token stride), the same
    pools, table, gate, split and packing: BITWISE (the block hands the adapter exactly these views).  At 32 / 2, ``S_q = 2``: the
    32-column decode tile in one unit (``S_q = 3`` rides the prefill route under the default policy)."""
    case = _std_case(16, 2)
    case.run()
    case.check_against_oracle()
    _, _, o_blk, _ = case.slab_views()
    api, o_buf, q, gate_v = _adapter_like(case, sample_k=case.k_cache, sample_v=case.v_cache, paged=True, gate=True, split=case.blk.decode_split_kv)
    assert api.kernel_template == DECODE_TILE and api.split_kv == case.blk.decode_split_kv >= 2
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device=_DEV)
    api.execute(
        q.transpose(1, 2),
        case.k_cache,
        case.v_cache,
        o_buf.transpose(1, 2),
        seq_kv_lens=case.kv_lens_t,
        block_table=case.block_table,
        workspace=ws,
        gate=gate_v.transpose(1, 2),
    )
    torch.cuda.synchronize()
    assert torch.equal(o_buf, o_blk), f"block O != adapter O: max|diff| {(o_buf.float() - o_blk.float()).abs().max().item():.3e}"


@requires_rubin
def test_paged_equals_dense_on_the_same_tokens_bitwise():
    """QI00's claim on the block's OWN operands: after the DECODE form ran (the step's tokens written through), the standalone
    adapter over the pools and over a padded DENSE cache holding the same tokens (gathered from the pools, padded to whole
    tiles), the same slab Q views, the same packing, both UNSPLIT and ungated (the padded dense adapter has no split; the
    gate rides the combine of a split) -- O and LSE BITWISE.  The block's own split plan is bitwise the standalone paged
    adapter at that split (test_the_block_form_equals_the_standalone_adapter_bitwise_on_the_same_views), and split ==
    unsplit within the combine's budget (test_split_equals_unsplit_within_the_combine_rounding): the three cells chain."""
    case = _std_case(12, 2)
    case.run()
    case.check_against_oracle()
    B, h_kv, P = case.blk.batch, _GROUPS[12][1], case.page
    S = -(-case.max_kv_len // _SPLIT) * _SPLIT
    k_dense = torch.zeros(B, S, h_kv, _D, dtype=case.dtype, device=_DEV)
    v_dense = torch.zeros_like(k_dense)
    for b, L in enumerate(case.kv_lens):
        for p in range(L):
            pg = int(case.block_table[b, p // P])
            k_dense[b, p] = case.k_cache[pg, :, p % P]
            v_dense[b, p] = case.v_cache[pg, :, p % P]
    assert torch.isfinite(k_dense.float()).all() and (k_dense.float().abs() < _SENTINEL).all(), "the live rows are the written tokens, never the sentinel"
    o_paged, lse_paged, api_p = _pre_gate_o(case, 1)
    api_d, o_dense, q, _ = _adapter_like(case, sample_k=k_dense.transpose(1, 2), sample_v=v_dense.transpose(1, 2), paged=False, gate=False, split=1, lse=True)
    assert api_p.kernel_template == api_d.kernel_template == DECODE_TILE and api_p.split_kv == api_d.split_kv == 1
    lse_dense = torch.empty_like(lse_paged)
    ws = torch.empty(max(api_d.scratch_workspace_bytes(), 1), dtype=torch.uint8, device=_DEV)
    api_d.execute(
        q.transpose(1, 2),
        k_dense.transpose(1, 2),
        v_dense.transpose(1, 2),
        o_dense.transpose(1, 2),
        lse_tensor=lse_dense,
        seq_kv_lens=case.kv_lens_t,
        workspace=ws,
    )
    torch.cuda.synchronize()
    assert torch.equal(o_paged, o_dense), f"paged O != dense O: max|diff| {(o_paged.float() - o_dense.float()).abs().max().item():.3e}"
    assert torch.equal(lse_paged, lse_dense), f"paged LSE != dense LSE: max|diff| {(lse_paged - lse_dense).abs().max().item():.3e}"


def _ulp(mag, mbits):
    """The output dtype's ulp at each element's binade, floored at the subnormal step (the decode suite's helper)."""
    _, exp = torch.frexp(mag.cpu())
    emin = {7: -126, 10: -14}[mbits]
    ulp = torch.ldexp(torch.ones_like(mag.cpu()), exp - 1 - mbits)
    return torch.clamp(ulp, min=2.0 ** (emin - mbits)).to(mag.device)


def _pre_gate_o(case, split):
    """The pre-gate O of the block's own stage (4) at ``split`` on the block's post-execute slab views: a fresh ungated stage
    declared like the block's (the gate OFF: the combine applies none), executed into its own buffer."""
    api, o_buf, q, _ = _adapter_like(case, sample_k=case.k_cache, sample_v=case.v_cache, paged=True, gate=False, split=split, lse=True)
    lse = torch.empty(case.blk.batch, case.geom.h_q, case.blk.seq_len, dtype=torch.float32, device=_DEV)
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device=_DEV)
    api.execute(
        q.transpose(1, 2),
        case.k_cache,
        case.v_cache,
        o_buf.transpose(1, 2),
        lse_tensor=lse,
        seq_kv_lens=case.kv_lens_t,
        block_table=case.block_table,
        workspace=ws,
    )
    torch.cuda.synchronize()
    return o_buf, lse, api


def _split_budget(case, a, b, q_rows):
    """The decode suite's two-term budget between a split and an unsplit plan, per O element, generalized to S_q >= 1 with
    the bottom-right band: ``|a - b| <= ulp(max(|a|, |b|)) + 2 eps_P sum_j p_j |v_j| / l`` -- one output rounding of two
    differently associated fp32 sums, plus twice the half-precision P quantization each path does at its own running max,
    weighted by the row's softmax-mean |V| (computed from the same pools in fp32)."""
    mbits = {torch.bfloat16: 7, torch.float16: 10}[case.dtype]
    eps_p = 2.0 ** -(mbits + 1)
    B, s_q, H, d = q_rows.shape
    KH, P = _GROUPS[case.group][1], case.page
    G = H // KH
    budget = torch.zeros(B, s_q, H, d, dtype=torch.float64, device=_DEV)
    live = torch.zeros(B, s_q, H, dtype=torch.bool, device=_DEV)
    for bi, L in enumerate(case.kv_lens):
        if L == 0:
            continue
        pages = case.block_table[bi, : -(-L // P)].tolist()
        k = torch.cat([case.k_cache[pg] for pg in pages], dim=1)[:, :L].float()  # [KH, L, d]
        v = torch.cat([case.v_cache[pg] for pg in pages], dim=1)[:, :L].float().abs()
        kv_h = torch.arange(H, device=_DEV) // G
        qf = q_rows[bi].float()  # [s_q, H, d]
        scores = torch.einsum("shd,hLd->shL", qf, k[kv_h]) * case.geom.scale  # [s_q, H, L]
        keys = torch.arange(L, device=_DEV)
        allowed = keys[None, :] <= (torch.arange(s_q, device=_DEV) + (L - s_q))[:, None]  # bottom-right
        scores = scores.masked_fill(~allowed[:, None, :], float("-inf"))
        prob = torch.softmax(scores, dim=-1)
        pv_abs = torch.einsum("shL,hLd->shd", prob.nan_to_num(0.0), v[kv_h])
        budget[bi] = (2.0 * eps_p) * pv_abs.double()
        live[bi] = torch.isfinite(prob).all(dim=-1)
    a64, b64 = a.double(), b.double()
    mag = torch.maximum(a64.abs(), b64.abs())
    budget = budget + _ulp(mag, mbits)
    diff = (a64 - b64).abs()
    m = live.unsqueeze(-1).expand_as(diff)
    return diff[m], budget[m]


@requires_rubin
@pytest.mark.parametrize("group", [16, 8])
def test_split_equals_unsplit_within_the_combine_rounding(group):
    """``decode_split_kv=1`` (the tile unsplit) vs a four-way split on the same step, the gate OUT of both (the
    pre-gate O of the block's own stage): not bitwise, bounded by the derived two-term budget; the fp32 LSEs agree to fp32
    rounding; ``out`` of the two blocks within the dense suite's budget of each other and of the oracle."""
    lens = (1000, 384, 0, 2048)
    # the split PINNED at 4: the eager decode split model keeps a 4-unit x 16-tile launch unsplit (the second host launch costs
    # more than it saves there), and this cell is about the two plans' numerics, not the model's choice (pinned in
    # test_the_split_choice_is_the_decode_split_model_floored_at_two_under_the_gate)
    split_case = _case(("split", group, "four"), group, 2, lens, seed=11, blk_kw=dict(fuse_gate=False, decode_split_kv=4))
    unsplit_case = _case(("split", group, "one"), group, 2, lens, seed=11, blk_kw=dict(fuse_gate=False, decode_split_kv=1))
    assert split_case.blk.decode_split_kv == 4 and unsplit_case.blk.decode_split_kv == 1 and not split_case.blk.gate_fused and not unsplit_case.blk.gate_fused
    out_s, out_u = split_case.run().clone(), unsplit_case.run().clone()
    split_case.check_against_oracle(what=f"split {split_case.blk.decode_split_kv}")
    unsplit_case.check_against_oracle(what="unsplit")
    q_s, _, _, _ = split_case.slab_views()
    q_u, _, _, _ = unsplit_case.slab_views()
    assert torch.equal(q_s, q_u), "the two blocks must see the same Q (same inputs, same projection)"
    o_s, lse_s, api_s = _pre_gate_o(split_case, split_case.blk.decode_split_kv)
    o_u, lse_u, api_u = _pre_gate_o(unsplit_case, 1)
    assert api_s.kernel_template == api_u.kernel_template == DECODE_TILE and api_s.split_kv == 4 and api_u.split_kv == 1
    diff, budget = _split_budget(split_case, o_s, o_u, q_s)
    ratio = (diff / budget).max().item()
    print(
        f"[{group}] split {split_case.blk.decode_split_kv} vs unsplit pre-gate O: max |diff| {diff.max().item():.3e}, {(diff > 0).double().mean().item():.4%} of the live "
        f"elements differ, max |diff| / budget {ratio:.3f}"
    )
    assert ratio <= 1.0, ratio
    live = torch.isfinite(lse_u)
    lse_diff = (lse_s[live] - lse_u[live]).abs().max().item()
    print(f"[{group}] split vs unsplit LSE: max |diff| {lse_diff:.3e} on |LSE| up to {lse_u[live].abs().max().item():.2f}")
    torch.testing.assert_close(lse_s[live], lse_u[live], atol=1e-5, rtol=4e-6)
    assert torch.isneginf(lse_u[~live]).all() and torch.isneginf(lse_s[~live]).all()
    c = _cos(out_s[split_case.live], out_u[split_case.live])
    print(f"[{group}] split vs unsplit out: cos {c:.6f} max |diff| {(out_s.float() - out_u.float()).abs().max().item():.3e}")
    assert c > 0.999


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_the_gate_in_the_combine_vs_the_separate_gate_launch_within_one_rounding(dtype):
    """``fuse_gate=True`` (the gate applied to the merged fp32 value in the split COMBINE, one rounding) vs ``fuse_gate=False``
    at the SAME split (the tile's bf16 / f16 O, then stage (5): ``round(O * sigmoid(G))`` -- two roundings and its own approx
    sigmoid), per element of O: ``|a - r| <= ulp(max(|a|, |r|)) + sigmoid(g) ulp(|b|) / 2 + |b| 2^-11`` with ``b`` the pre-gate
    O (the decode suite's stated budget: the two final roundings, the intermediate rounding the unfused path carries, and the
    approximate tanh behind both sigmoids).  The exact max |diff| is the record; ``out`` of both within the suite's budget."""
    lens = (1000, 0, 384, 77)
    fused = _case(("gate", dtype, "combine"), 16, 2, lens, dtype=dtype, seed=19)
    split = fused.blk.decode_split_kv
    unfused = _case(("gate", dtype, "launch"), 16, 2, lens, dtype=dtype, seed=19, blk_kw=dict(fuse_gate=False, decode_split_kv=split))
    assert fused.blk.gate_fused and not unfused.blk.gate_fused and unfused.blk.decode_split_kv == split >= 2
    out_f, out_u = fused.run().clone(), unfused.run().clone()
    fused.check_against_oracle(what="gate-in-combine")
    unfused.check_against_oracle(what="gate launch")
    q_f, gate_f, o_f, _ = fused.slab_views()
    q_u, gate_u, o_u, _ = unfused.slab_views()
    assert torch.equal(q_f, q_u) and torch.equal(gate_f, gate_u)
    b, _, _ = _pre_gate_o(unfused, split)  # the ungated merged O the separate launch gated
    mbits = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    a, r, b64 = o_f.double(), o_u.double(), b.double()
    s = torch.sigmoid(gate_f.double())
    live = fused.live.view(-1, 1, 1, 1).expand_as(a)
    budget = _ulp(torch.maximum(a.abs(), r.abs()), mbits) + s * _ulp(b64.abs(), mbits) * 0.5 + b64.abs() * 2.0**-11
    diff = (a - r).abs()
    ratio = (diff[live] / budget[live]).max().item()
    print(
        f"[{dtype}] gate-in-combine vs the separate gate launch (split {split}): max |diff| {diff[live].max().item():.3e} on |O| up to "
        f"{r[live].abs().max().item():.3f}, {(diff[live] > 0).double().mean().item():.4%} of the live elements differ, max |diff| / budget {ratio:.3f}; "
        f"out: cos {_cos(out_f[fused.live], out_u[fused.live]):.6f} max |diff| {(out_f.float() - out_u.float()).abs().max().item():.3e}"
    )
    assert ratio <= 1.0, ratio
    assert (a[~live] == 0).all() and (r[~live] == 0).all(), "the padded entry: exactly 0 on both"
    assert _cos(out_f[fused.live], out_u[fused.live]) > 0.999


@requires_rubin
def test_get_workspace_size_is_honest_and_holds_the_split_partials():
    """The split partials (the SDPA's scratch) sit inside ``get_workspace_size()`` (the engine region = the max of the GEMMs'
    and the SDPA's scratch, behind the carve), and nothing past the size is written: a sentinel tail survives an execute."""
    case = _std_case(16, 2)
    blk = case.blk
    lay = blk._layout()
    scratch = blk._sdpa.scratch_workspace_bytes()
    assert blk.decode_split_kv >= 2 and scratch > 0, (blk.decode_split_kv, scratch)
    assert blk.get_workspace_size() >= lay.total_bytes + scratch
    need = blk.get_workspace_size()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    out = torch.empty_like(case.out)
    case.execute(out=out, ws=ws)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.check_against_oracle(out, what="sentinel-tailed workspace")


@requires_rubin
def test_two_launches_are_bitwise():
    case = _std_case(6, 2)
    a = case.run().clone()
    b = case.run().clone()
    assert torch.equal(a, b)


def _capture_replay_cell(quant=None, **blk_kw):
    """The serving stack's defaults: the FIRST execute of a fresh declaration captured on a NEW stream with no eager warm-up
    under ``set_sync_debug_mode("error")`` (no host sync in the DECODE path: lengths, tables and slots are device data); the
    capture launches nothing, the replay is BITWISE a fresh eager run; a padded batch replays correctly; new ``kv_lens`` /
    ``block_table`` / ``slot_mapping`` / ``h`` through the captured tensors replay bitwise the eager run over them.  ``quant``
    (appended; True = the FP8-projection + bf16-attention mode) hands the case the mode; the sibling step then reuses the captured
    block's scales (its codes must read under the captured spec)."""
    lens = (300, 0, 1000, 77)
    case = _Case(16, 2, lens, seed=23, blk_kw=blk_kw or None, quant=quant)  # fresh: never executed
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.empty(case.blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed = case.out.clone()
        case.check_against_oracle(replayed, what="replay")
        # eager over the same inputs: the pools already hold the step's tokens (idempotent rows), out bitwise
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed), "the replay of the captured first execute differs from the eager run"
        # a NEW step through the captured tensors: new lengths (one entry longer by a token: its slot moves), a re-laid-out
        # table (pages swapped between two entries), new slots and new h -- the oracle of the new step through a sibling case
        sib = _Case(16, 2, (301, 0, 1000, 77), seed=29, table_row_stride=case.block_table.shape[1], quant=case.scales if quant else None)
        # the sibling's pools hold its oracle prefix; make the CAPTURED pools hold them (the captured pointers are the block's)
        case.k_cache.copy_(sib.k_cache)
        case.v_cache.copy_(sib.v_cache)
        case.block_table.copy_(sib.block_table)
        case.kv_lens_t.copy_(sib.kv_lens_t)
        case.slot_mapping.copy_(sib.slot_mapping)
        case.h.copy_(sib.h)
        case.cos.copy_(sib.cos)
        case.sin.copy_(sib.sin)
        case.w["w_qkvg"].copy_(sib.w["w_qkvg"])
        case.w["w_o"].copy_(sib.w["w_o"])
        case.out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2 = case.out.clone()
        assert torch.isfinite(replayed2.float()).all() and (replayed2[1] == 0).all()
        c = _cos(replayed2[sib.live], sib.ref_out[sib.live])
        print(f"replay over new lengths / table / slots / inputs vs the new step's oracle: cos {c:.6f}")
        assert c > 0.999
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed2), "the replay over new lengths / tables / slots differs from the eager run over them"
    finally:
        graph.reset()


@requires_rubin
def test_capture_of_the_first_execute_replays_bitwise_and_takes_new_lengths_tables_slots_and_inputs():
    _capture_replay_cell()


@requires_rubin
def test_capture_of_the_first_execute_replays_bitwise_on_the_plain_chain_with_the_separate_cache_write():
    """The same cell on the chain as before (``use_pdl=False``, the separate cache-write launch).  The default cell above runs the PDL
    chain with the paged write-through folded into the norm + RoPE launch (the arm's slot loads and pool stores are device-side like the
    separate kernel's), so the two cells together capture both chains with no host sync and replay bitwise."""
    _capture_replay_cell(use_pdl=False, fuse_cache_write_small_m=False)


# ---------------------------------------------------------------------------
# The 64-column decode tile: the MTP step (S_q = 4) of the 24/2 and 32/2 groups in ONE token unit
# ---------------------------------------------------------------------------
#
# The swap-AB decode tile's third width (config_sm107._D256_DECODE_Q_TILES_SM107: 16 softmax warps, 576 threads, 259 KiB SMEM)
# takes 4 tokens x 12 / 16 heads = 48 / 64 packed rows in one unit, so the KV range is streamed ONCE per (sequence, KV head)
# where the 32-column tile cut the step into two token units.  ROUTED (the last entry of config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES):
# these cells run the PRODUCTION route; the two-unit 32-column route is the ``decode_q_tile_max=32`` arm (the route before the width
# landed, value for value), declared beside it for the bitwise pin.


def _w64_case(group, s_q=4, **kw):
    """The S_q = 4 cell of ``group`` on the production route -- the 64-column tile in ONE unit -- declared, compiled and memoized
    like the standard cells, the route pinned before anything compiles."""
    key = ("w64", group, s_q, repr(sorted(kw.items())))
    if key not in _CASES:
        case = _Case(group, s_q, _LENS, **kw)
        assert (case.blk.decode_route.n_q, case.blk.decode_route.token_units, case.blk.decode_q_tile_max) == (64, 1, None), case.blk.decode_route
        _CASES[key] = case.compile()
    return _CASES[key]


@requires_rubin
@pytest.mark.parametrize("group", [16, 12])
def test_decode_s_q_four_on_the_64_column_tile_matches_the_oracle(group):
    """The MTP step (S_q = 4) of the 32/2 (64 rows) and 24/2 (48 rows + 16 never-stored rows) groups on the 64-column tile in
    ONE token unit, against the per-sequence fp32 oracle (cos > 0.999 on ``out``), the padded entry EXACTLY 0 with block 0
    untouched, the route the block declared and the module the adapter compiled agreeing on the width (N_Q 64, the whole group
    per tile, one unit, the gate OUT of the kernel: in the combine)."""
    case = _w64_case(group)
    r, rb = case.blk.decode_route, case.route_readback()
    assert (r.n_q, r.token_units, r.pack_gqa, r.served, r.tile) == (64, 1, True, True, DECODE_TILE), r
    assert rb["template"] == DECODE_TILE and rb["file"].endswith("/sm107/decode_d256_f16.py"), rb
    assert (rb["n_q"], rb["heads_per_tile"], rb["adapter_pack"]) == (64, group, True), rb
    assert rb["epilogue_gate"] is False and case.blk.gate_fused and case.blk.decode_split_kv >= 2 and rb["split"] == case.blk.decode_split_kv, rb
    case.run()
    case.check_against_oracle()
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0), "block 0 (the NULL block) must never be written"


@requires_rubin
@pytest.mark.parametrize("group", [16, 12])
def test_decode_s_q_four_on_the_64_column_tile_equals_the_two_unit_route_bitwise(group):
    """The same step through the production route (the 64-column tile in one unit) and through the ``decode_q_tile_max=32`` arm
    (the 32-column tile in two token units: the route before the width landed), at the SAME split (pinned to 2 on both: the
    split's KV ranges do not depend on the token unit, so every row's partials are the same values and the combine sums them in
    the same order): the block's ``out`` is BITWISE equal -- the width moves rows between CTAs, it never changes the arithmetic
    (the SDPA suite's cross-form pin, at the block level); the cap reads back on the route and on the compiled module."""
    wide = _w64_case(group, blk_kw={"decode_split_kv": 2})
    # the cap alone sends 32 / 2 at S_q 4 (64 rows) to the PREFILL tile under the default policy; the two-unit 32-column route is the cap
    # under token_units (the adapter's route before the width landed) -- at 24 / 2 the policies coincide
    narrow = _case(
        ("s2cap32", group, 4), group, 4, _LENS, blk_kw={"decode_split_kv": 2, "decode_q_tile_max": 32, "decode_route_policy": DECODE_ROUTE_TOKEN_UNITS}
    )
    assert (wide.blk.decode_route.n_q, wide.blk.decode_route.token_units) == (64, 1) and (narrow.blk.decode_route.n_q, narrow.blk.decode_route.token_units) == (
        32,
        2,
    )
    assert (narrow.blk.decode_q_tile_max, narrow.blk.decode_route.q_tile_max, narrow.route_readback()["n_q"], wide.route_readback()["n_q"]) == (32, 32, 32, 64)
    assert wide.blk.decode_split_kv == narrow.blk.decode_split_kv == 2
    a = wide.run().clone()
    b = narrow.run().clone()
    wide.check_against_oracle(a, what="64-column one unit")
    narrow.check_against_oracle(b, what="32-column two units")
    assert torch.equal(a, b), f"64-column one unit != 32-column two units: max |diff| {(a.float() - b.float()).abs().max().item():.3e}"


@requires_rubin
def test_decode_s_q_four_on_the_64_column_tile_capture_of_the_first_execute_replays_bitwise_and_takes_new_lengths():
    """The serving stack's defaults on the 64-column tile: the FIRST execute of a fresh S_q = 4 declaration (32/2, one 64-row
    unit) captured on a NEW stream with no eager warm-up under ``set_sync_debug_mode("error")``; the capture launches nothing,
    the replay is BITWISE a fresh eager run; a padded batch replays correctly; new ``kv_lens`` / ``block_table`` /
    ``slot_mapping`` / ``h`` through the captured tensors replay bitwise the eager run over them."""
    lens = (300, 0, 1000, 77)
    case = _Case(16, 4, lens, seed=23)  # fresh: never executed
    sib = _Case(16, 4, (301, 0, 1000, 77), seed=29, table_row_stride=case.block_table.shape[1])
    assert (case.blk.decode_route.n_q, case.blk.decode_route.token_units) == (64, 1)
    # The adapter reads the rule again at compile (its own _decode_q_tile); the stage pins the compiled WIDTH against the route
    # there (a drift the kernel_template pin cannot see: both widths are decode_d256_f16), and the readback repeats it here.
    case.blk.check_support()
    case.blk.compile()
    assert case.route_readback()["n_q"] == 64, case.route_readback()
    case.ws = torch.empty(case.blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed = case.out.clone()
        case.check_against_oracle(replayed, what="replay (64-column tile)")
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed), "the replay of the captured first execute differs from the eager run"
        # a NEW step through the captured tensors (the sibling's oracle prefix, a table with pages swapped, new slots / h)
        assert sib.block_table.shape[1] == case.block_table.shape[1]
        case.k_cache.copy_(sib.k_cache)
        case.v_cache.copy_(sib.v_cache)
        case.block_table.copy_(sib.block_table)
        case.kv_lens_t.copy_(sib.kv_lens_t)
        case.slot_mapping.copy_(sib.slot_mapping)
        case.h.copy_(sib.h)
        case.cos.copy_(sib.cos)
        case.sin.copy_(sib.sin)
        case.w["w_qkvg"].copy_(sib.w["w_qkvg"])
        case.w["w_o"].copy_(sib.w["w_o"])
        case.out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2 = case.out.clone()
        assert torch.isfinite(replayed2.float()).all() and (replayed2[1] == 0).all()
        c = _cos(replayed2[sib.live], sib.ref_out[sib.live])
        print(f"64-column tile: replay over new lengths / table / slots / inputs vs the new step's oracle: cos {c:.6f}")
        assert c > 0.999
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed2), "the replay over new lengths / tables / slots differs from the eager run over them"
    finally:
        graph.reset()


# --- the split ladder on the decode-grid combine ---------------------------------------------------------------------------


def _ladder_lens(batch):
    # kv 4K at B 1; at B 4 three live lengths beside it (a tail tile, a page-unaligned one) and a PADDED entry (0: block 0 + slot -1)
    return [4096] if batch == 1 else [4096, 3000, 2048, 0]


@requires_rubin
@pytest.mark.parametrize("split", [4, 8, 16], ids=lambda s: f"split{s}")
@pytest.mark.parametrize("s_q", [1, 4], ids=["sq1", "sq4"])
@pytest.mark.parametrize("batch", [1, 4], ids=["b1", "b4"])
def test_decode_split_ladder_recombines_on_the_decode_grid(batch, s_q, split):
    """The serving step (group 12 = 24/2, page 16, the gate in the COMBINE, the MTP step in token units) at a PINNED split -- the
    splits the kv 4K route offers (its 32 KV tiles bound the candidates at 16) -- against the per-sequence fp32 oracle: every cell's
    combine is the shared pass's DECODE-GRID kernel (one block per (row, head, batch, 64-column chunk): the lean kernel's grid of
    ``ceil(S_q / 4) x 24 x B`` blocks is under ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS``), the padded entry exactly 0, the route
    readback at the pinned split with the gate OUT of the tile."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert comb.decode_grid_selected(batch, 24, s_q, _D)
    lens = _ladder_lens(batch)
    case = _case(("ladder", batch, s_q, split), 12, s_q, lens, seed=31 + split, blk_kw=dict(fuse_gate=True, decode_split_kv=split))
    assert case.blk.decode_split_kv == split and case.blk.gate_fused
    case.run()
    case.check_against_oracle(what=f"b{batch} sq{s_q} split {split}")
    rb = case.route_readback()
    assert rb["template"] == case.blk.decode_route.tile and rb["split"] == split and rb["epilogue_gate"] is False, rb
    if 0 in lens:
        assert (case.out[lens.index(0)] == 0).all(), "the padded entry writes exactly 0"


# ---------------------------------------------------------------------------
# The paged write-through folded into the norm + RoPE launch (fuse_cache_write_small_m): one launch fewer, the pools bitwise
# ---------------------------------------------------------------------------
#
# kernels/qk_norm_rope_tma.py's pool-write arm: the slab's V band rides the launch as a third tile band and the post-RoPE K / V rows go
# to the pools from the same tiles (the K words the slab's own, the V words a bit copy); the separate cache-write stage is not built.
# The control is the chain as measured (the separate launch).  The two must leave BITWISE pools, slab bands and `out` on every pool
# layout (the three served stride patterns + an HND pool with a padded page stride), both slot widths, with a padded entry (slot -1)
# and with a slot pushed past the pools' capacity (a contract violation neither arm may write).  RED on a tree without the arm: the
# declaration knob does not exist there (TypeError), and the V band is not part of the norm + RoPE launch.

_POOL_LAYOUTS_PW = _LAYOUTS + ("hnd_padded",)
_PW_LENS = (200, 0, 77, 1000)  # three live entries (a tail page each) around a PADDED one (block 0, slot -1)


from gated_block_census import cupti_kernels as _cupti_kernels  # the shared census helper (moved out: no test-module import for it)


def _slab_bands(case):
    """The slab's Q, GATE, K and V bands after an execute, as the block's ``[T, H, D]`` column views (K and Q post-norm + RoPE)."""
    g, lay = case.geom, case.blk._ws
    t = case.blk.batch * case.blk.seq_len
    proj = _view(case.ws, lay.proj, (t, g.n_qkvg), case.dtype)
    o_q, o_g, o_k, o_v = g.qkvg_offsets[:4]
    return tuple(_cols(proj, off, h, g.d_head) for off, h in ((o_q, g.h_q), (o_g, g.h_q), (o_k, g.h_kv), (o_v, g.h_kv)))


def _pw_pair(layout, *, group=12, s_q=2, page=16, seed=3, dtype=torch.bfloat16):
    """The control (the separate cache-write launch) and the arm (the write-through in the norm + RoPE launch) of one cell: identical
    inputs, prefixes and pools, compiled once per layout (the stage compiles BOTH slot widths); each keeps a pristine copy of its pools
    (the oracle prefix + the sentinel) to restore before every run."""
    key = ("pw", layout, group, s_q, page, seed, dtype)
    if key not in _CASES:
        pair = []
        for arm in (False, True):
            c = _Case(group, s_q, _PW_LENS, page=page, layout=layout, seed=seed, dtype=dtype, blk_kw=dict(fuse_cache_write_small_m=arm)).compile()
            c.pristine = (c.k_cache.clone(), c.v_cache.clone())
            pair.append(c)
        _CASES[key] = tuple(pair)
    return _CASES[key]


def _pw_run(case, slot_mapping):
    case.k_cache.copy_(case.pristine[0])
    case.v_cache.copy_(case.pristine[1])
    case.out.fill_(float("nan"))
    case.execute(slot_mapping=slot_mapping)
    torch.cuda.synchronize()


@requires_rubin
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("layout", _POOL_LAYOUTS_PW)
def test_the_write_through_in_the_norm_rope_launch_leaves_bitwise_pools_bands_and_out(layout, slot_dtype):
    """The pools (K and V), the slab's four bands and ``out`` of the arm are BITWISE the separate launch's on ``layout`` with ``slot_dtype``
    slots: (i) the padded cell (three live entries, one padded: slot -1 writes nothing, block 0 untouched, ``out`` vs the oracle), (ii) the
    same cell with one live token's slot past BOTH pools' capacity (neither arm writes it: exactly ``H_kv`` rows fewer per pool)."""
    ctl, arm = _pw_pair(layout)
    h_kv = _GROUPS[12][1]
    assert arm.blk.cache_write_in_norm_rope is True and arm.blk._cache_write is None and "cache_write" not in [st.name for st in arm.blk._stages]
    assert ctl.blk.cache_write_in_norm_rope is False and isinstance(ctl.blk._cache_write, _CacheWrite)
    assert arm.blk._norm_rope.page_size == 16 and ctl.blk._norm_rope.page_size == 0 and arm.blk._norm_rope._impl == "tma"
    slots = ctl.slot_mapping.to(slot_dtype)
    oor = slots.clone()
    first_live = int(torch.nonzero(slots >= 0)[0].item())
    oor[first_live] = ctl.n_pages * ctl.page + 3  # past both pools' capacity (n_pages x page slots): a contract violation, never written
    written = {}
    for tag, sm in (("padded", slots), ("out-of-range", oor)):
        for c in (ctl, arm):
            _pw_run(c, sm)
        what = f"{layout} {slot_dtype} {tag}"
        for name, a, b in (("K pool", arm.k_cache, ctl.k_cache), ("V pool", arm.v_cache, ctl.v_cache)):
            assert torch.equal(_bits(a), _bits(b)), f"{what}: the {name} of the arm differs from the separate launch's"
        assert torch.equal(_bits(arm.out), _bits(ctl.out)), f"{what}: out differs"
        for name, a, b in zip(("Q", "GATE", "K", "V"), _slab_bands(arm), _slab_bands(ctl)):
            assert torch.equal(_bits(a), _bits(b)), f"{what}: the slab's {name} band differs"
        k0, v0 = arm.block0_bits
        assert torch.equal(_bits(arm.k_cache[0]), k0) and torch.equal(_bits(arm.v_cache[0]), v0), f"{what}: block 0 written"
        written[tag] = (_written(arm.k_cache), _written(arm.v_cache))
        if tag == "padded":
            arm.check_against_oracle(what=f"{what} (arm)")
            assert written[tag][0] == _written(ctl.pristine[0]) + sum(2 for L in _PW_LENS if L) * h_kv, "the step's rows alone were written"
    assert written["out-of-range"] == tuple(w - h_kv for w in written["padded"]), "the out-of-range token's rows must not land anywhere"


@requires_rubin
def test_the_write_through_in_the_norm_rope_launch_serves_f16_pools_bitwise():
    """f16 activations (the unfused chain by declaration): the arm's pools and ``out`` bitwise the separate launch's on the compact layout."""
    ctl, arm = _pw_pair("hnd", dtype=torch.float16, seed=5)
    for c in (ctl, arm):
        _pw_run(c, c.slot_mapping)
    assert (
        torch.equal(_bits(arm.k_cache), _bits(ctl.k_cache))
        and torch.equal(_bits(arm.v_cache), _bits(ctl.v_cache))
        and torch.equal(_bits(arm.out), _bits(ctl.out))
    )
    arm.check_against_oracle(what="f16 arm")


@requires_rubin
def test_the_write_through_in_the_norm_rope_launch_drops_the_cache_write_launch_seven_to_six():
    """The DECODE chain's CUPTI census at the 24/2 shard's geometry (d_model 2560, B 1 x KV 4096, S_q 1, use_pdl): the separate-launch
    chain is 7 kernels (slab GEMM, norm + RoPE, cache write, decode tile, split combine, out-proj GEMM, split-K reduce); with the
    write-through in the norm + RoPE launch it is 6 -- the dropped name is the cache write's, `out` bitwise.  The pin names THIS tree's
    chain (the two-kernel split-K reduce); a tree with the in-GEMM split-K reduce reads 6 / 5."""
    cells = {}
    for arm in (False, True):
        cells[arm] = _Case(12, 1, (4096,), seed=1, d_model=2560, blk_kw=dict(use_pdl=True, fuse_cache_write_small_m=arm)).compile()
    k_off, m_off = _cupti_kernels(lambda: cells[False].execute())
    k_on, m_on = _cupti_kernels(lambda: cells[True].execute())
    if k_off is None or k_on is None:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    print(f"\nseparate launch: {len(k_off)} kernels + {len(m_off)} memset/memcpy:\n  " + "\n  ".join(n[:120] for n in k_off))
    print(f"in the norm + RoPE launch: {len(k_on)} kernels + {len(m_on)} memset/memcpy:\n  " + "\n  ".join(n[:120] for n in k_on))
    assert any("frost_cache_write" in n for n in k_off) and not any("frost_cache_write" in n for n in k_on)
    assert any("frost_qk_norm_rope_tma" in n for n in k_on)
    assert not [n for n in m_on if "memcpy" in n.lower()], "a hidden copy on the arm's execute path"
    assert (len(k_off), len(k_on)) == (7, 6), (len(k_off), len(k_on))
    torch.cuda.synchronize()
    cells[True].check_against_oracle(what="six launches")
    assert torch.equal(_bits(cells[True].out), _bits(cells[False].out))
    assert torch.equal(_bits(cells[True].k_cache), _bits(cells[False].k_cache)) and torch.equal(_bits(cells[True].v_cache), _bits(cells[False].v_cache))


def test_fuse_cache_write_small_m_resolves_the_measured_rule_by_default_and_the_arm_owns_the_write(monkeypatch):
    _rubin_reading(monkeypatch)
    """``fuse_cache_write_small_m`` (appended; None = AUTO): where the paged write-through runs on the DECODE form's unfused chain.  AUTO is
    the MEASURED rule (``CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS``): the write-through folds into the norm + RoPE launch (the stage carries the
    page size, the separate stage is not built, the stage list loses ``cache_write``) at ``B x S_q < 128`` rows and at ``S_q = 1``
    (measured to B 128; past it only under an explicit ``fuse_norm_rope=False``); the separate launch runs at ``B x S_q >= 128`` with
    ``S_q > 1`` (where the arm measured slower) -- both edges pinned, an explicit
    True / False winning on either side; the C0-lowered ``fuse_norm_rope=True`` declaration follows the same rule where it lowers and leaves
    the fork its own write-through past the bound; f16 pools take the arm; the typed declines (the fused fork, a self block, a non-bool); the
    carve does not read it."""
    from cudnn.gated_attention_block import CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS

    assert CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS == 128
    at = lambda b=1, s_q=1, **kw: _make_block(12, b, s_q, **kw)[0]  # noqa: E731
    names = lambda blk: [st.name for st in blk._stages]  # noqa: E731
    auto = at()
    assert auto.fuse_cache_write_small_m is None and auto.cache_write_in_norm_rope is True and auto._cache_write is None
    assert auto._norm_rope.page_size == 16 and "cache_write" not in names(auto) and "qk_norm_rope" in names(auto)
    sep = at(fuse_cache_write_small_m=False)
    assert sep.fuse_cache_write_small_m is False and sep.cache_write_in_norm_rope is False and isinstance(sep._cache_write, _CacheWrite)
    assert sep._norm_rope.page_size == 0 and "cache_write" in names(sep)
    assert names(auto) == [n for n in names(sep) if n != "cache_write"], "the arm drops exactly the cache-write stage"
    on = at(fuse_cache_write_small_m=True)
    assert on.fuse_cache_write_small_m is True and on.cache_write_in_norm_rope is True and on._cache_write is None and names(on) == names(auto)
    # the measured rule at its edges on the unfused chain (fuse_norm_rope=False: declared unfused at any row count, so the rule alone
    # decides): every B 1 .. 4 cell and S_q = 1 (to B 129 here; measured to B 128) fold in; 128 rows with S_q > 1 do not (B 32 x 4 = the measured
    # loss), 127 do
    cells = (
        ((1, 1), True),
        ((4, 4), True),
        ((1, 32), True),
        ((32, 1), True),
        ((127, 1), True),
        ((128, 1), True),
        ((129, 1), True),
        ((31, 4), True),
        ((63, 2), True),
        ((32, 4), False),
        ((64, 2), False),
        ((16, 8), False),
        ((128, 4), False),
    )
    for (b, s_q), want in cells:
        blk = at(b, s_q, fuse_norm_rope=False)
        assert blk.fuse_cache_write_small_m is None and blk.cache_write_in_norm_rope is want and (blk._cache_write is None) is want, (b, s_q)
        assert ("cache_write" in names(blk)) is (not want) and blk._norm_rope.page_size == (16 if want else 0), (b, s_q)
        # an explicit value wins on either side of the rule
        assert at(b, s_q, fuse_norm_rope=False, fuse_cache_write_small_m=True).cache_write_in_norm_rope is True, (b, s_q)
        assert at(b, s_q, fuse_norm_rope=False, fuse_cache_write_small_m=False).cache_write_in_norm_rope is False, (b, s_q)
    # the default serving declaration (fuse_norm_rope=True): the C0 lowering runs the unfused chain to 128 rows -- the rule applies there
    # (16 rows fold in, 128 rows with S_q 4 keep the separate launch) -- and the fused fork past it, which owns the write-through itself
    low = at(4, 4, fuse_norm_rope=True)
    assert low.small_m_decode and not low.fuse_norm_rope_effective and low.cache_write_in_norm_rope and low._cache_write is None
    edge = at(32, 4, fuse_norm_rope=True)
    assert edge.small_m_decode and not edge.fuse_norm_rope_effective and not edge.cache_write_in_norm_rope and isinstance(edge._cache_write, _CacheWrite)
    high = at(129, 1, fuse_norm_rope=True)
    assert (
        not high.small_m_decode and high.fuse_norm_rope_effective and high.cache_write_fused and not high.cache_write_in_norm_rope and high._norm_rope is None
    )
    assert at(dtype=torch.float16).cache_write_in_norm_rope is True
    assert auto._layout() == sep._layout(), "the carve does not read the knob"
    # the typed declines
    with pytest.raises(TypeError, match="fuse_cache_write_small_m must be None"):
        at(fuse_cache_write_small_m=1)
    with pytest.raises(NotImplementedError, match="fused projection fork"):
        at(fuse_norm_rope=True, fuse_norm_rope_small_m=True, fuse_cache_write_small_m=True)
    # under AUTO the fork keeps its own write-through (the arm reads False, never a decline)
    fork = at(fuse_norm_rope=True, fuse_norm_rope_small_m=True)
    assert fork.cache_write_in_norm_rope is False and fork.cache_write_fused and fork._norm_rope is None
    geom, _ = _geom(12)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    with pytest.raises(NotImplementedError, match="needs paged_kv_page_size > 0"):
        GatedAttentionBlockFwd(
            mk(1, 1, _D_MODEL),
            mk(geom.n_qkvg, _D_MODEL),
            mk(_D),
            mk(_D),
            mk(1, 1, _ROPE),
            mk(1, 1, _ROPE),
            mk(_D_MODEL, 24 * _D),
            mk(1, 1, _D_MODEL),
            geom,
            fuse_cache_write_small_m=True,
        )


@contextlib.contextmanager
def monkeypatch_rubin_route():
    """The Rubin reading of the route for one declaration (the ``_rubin_reading`` fixture's effect without a monkeypatch argument)."""
    prev = _api._route_device_cc
    _api._route_device_cc = lambda device: None
    try:
        yield
    finally:
        _api._route_device_cc = prev


def test_fuse_cache_write_small_m_is_a_typed_decline_off_the_arm_and_auto_reads_false_there():
    """``fuse_cache_write_small_m=True`` off the arm's domain is a typed ``NotImplementedError`` naming the reason, and ``None`` reads False
    there with the separate cache-write stage built (where there is a write-through at all): the EXTEND form and the write-through on the
    block's own K / V (no dense decode route), the block-sparse decode mode -- a QsaSpec geometry with and without the indexer band, the
    e4m3 cache -- (no dense route either: the no-route row speaks for them), a QuantSpec pipeline (the write-through's own quant decline
    speaks first; without the attribute there is nothing to fold), fp32 activations on the dense form (the activation row).  The
    resolver's quant / indexer-band / e4m3 rows are GUARDS the constructor cannot reach today; their text is pinned on the resolver
    directly, so a form that admits one of those pipelines onto a decode route meets a pinned decline rather than a silent fold."""
    import test_block_qsa_decode as _qsa
    import test_block_qsa_fp8_kv as _qkv
    import test_block_serving_extend as _ext
    from cudnn.gated_attention_block import QuantSpec

    no_route = "is the dense DECODE form's arm"
    # the EXTEND form (packed, over the pools) and the dense write-through on the block's own K / V
    with pytest.raises(NotImplementedError, match=no_route):
        _ext._declare_extend(2, 128, max_kv_len=128, fuse_cache_write_small_m=True)
    ext_auto, *_ = _ext._declare_extend(2, 128, max_kv_len=128)
    assert ext_auto.cache_write_in_norm_rope is False and isinstance(ext_auto._cache_write, _CacheWrite) and ext_auto._norm_rope.page_size == 0
    geom, _ = _geom(12)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    args = (mk(1, 1, _D_MODEL), mk(geom.n_qkvg, _D_MODEL), mk(_D), mk(_D), mk(1, 1, _ROPE), mk(1, 1, _ROPE), mk(_D_MODEL, 24 * _D), mk(1, 1, _D_MODEL), geom)
    with pytest.raises(NotImplementedError, match=no_route):
        GatedAttentionBlockFwd(*args, paged_kv_page_size=16, fuse_cache_write_small_m=True)
    own = GatedAttentionBlockFwd(*args, paged_kv_page_size=16)
    assert own.kv_source == KV_SOURCE_SELF and own.cache_write_in_norm_rope is False and isinstance(own._cache_write, _CacheWrite)
    # the block-sparse decode mode: a QsaSpec geometry, the indexer band, the e4m3 cache -- no dense route
    for decl in (
        lambda **kw: _qsa._declare(_qsa.GEOM_SMALL, 2, 1, n_pages=4, fuse_norm_rope=False, **kw)[0],
        lambda **kw: _qsa._declare(_qsa.GEOM_SMALL, 4, 1, n_pages=6, index_band=True, fuse_norm_rope=False, **kw)[0],
        lambda **kw: _qkv._declare(_qkv.GEOM_SMALL, 2, 1, fuse_norm_rope=False, **kw)[0],
    ):
        with pytest.raises(NotImplementedError, match=no_route):
            decl(fuse_cache_write_small_m=True)
        blk = decl()
        assert blk.decode_route is None and blk.cache_write_in_norm_rope is False and blk._cache_write is not None and blk._norm_rope.page_size == 0
    # a QuantSpec pipeline: the write-through's own decline speaks first (e4m3 pools are the cache-dtype attribute's); nothing to fold without it
    spec = QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)
    g8 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE)
    e4 = lambda *shape: torch.zeros(*shape, dtype=torch.float8_e4m3fn, device=_DEV)  # noqa: E731
    qargs = (
        e4(1, 256, _D_MODEL),
        e4(g8.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(1, 256, _ROPE),
        mk(1, 256, _ROPE),
        e4(_D_MODEL, 8 * _D),
        mk(1, 256, _D_MODEL),
        g8,
    )
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with quant=QuantSpec"):
        GatedAttentionBlockFwd(*qargs, quant=spec, paged_kv_page_size=16, fuse_cache_write_small_m=True)
    with pytest.raises(NotImplementedError, match="needs paged_kv_page_size > 0"):
        GatedAttentionBlockFwd(*qargs, quant=spec, fuse_cache_write_small_m=True)
    quant = GatedAttentionBlockFwd(*qargs, quant=spec)
    assert quant.cache_write_in_norm_rope is False and quant._cache_write is None and quant.paged_kv_page_size == 0
    # the FP8-projection + bf16-attention mode (QuantSpec(attention_dtype=torch.bfloat16)) on the DECODE form: the fold is ADMITTED (the bf16
    # chain's TMA norm + RoPE kernel on a bf16 slab) -- AUTO True on the fold's domain, an explicit True followed, an explicit False the separate
    # launch; the stage list is the bf16 chain's plus the one quantize launch
    mode = QuantSpec.bf16_attention(0.01, 0.02, 0.03, 4.0)
    kc8, vc8 = _pools(8, 2, 16, _D, "hnd", torch.bfloat16)
    geom_br = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)  # the paged forms' diagonal
    mode_args = (
        e4(4, 1, _D_MODEL),
        e4(geom_br.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(4, 1, _ROPE),
        mk(4, 1, _ROPE),
        e4(_D_MODEL, 8 * _D),
        mk(4, 1, _D_MODEL),
        geom_br,
    )
    mode_kw = dict(
        paged_kv_page_size=16, kv_source="paged", sample_k_cache=kc8, sample_v_cache=vc8, sample_block_table=torch.zeros(4, 8, dtype=torch.int32, device=_DEV)
    )
    with monkeypatch_rubin_route():
        auto_mode = GatedAttentionBlockFwd(*mode_args, quant=mode, fuse_gate=True, decode_split_kv=2, **mode_kw)  # the gate in the combine
        assert (
            auto_mode.attention_bf16 and auto_mode.cache_write_in_norm_rope is True and auto_mode._cache_write is None and auto_mode._norm_rope.page_size == 16
        )
        assert [st.name for st in auto_mode._stages] == ["qkv_gate_proj", "qk_norm_rope", "sdpa", "quantize_o", "out_proj"] and auto_mode.gate_fused
        on_mode = GatedAttentionBlockFwd(*mode_args, quant=mode, fuse_cache_write_small_m=True, **mode_kw)
        assert on_mode.cache_write_in_norm_rope is True and on_mode._cache_write is None
        off_mode = GatedAttentionBlockFwd(*mode_args, quant=mode, fuse_cache_write_small_m=False, fuse_gate=False, **mode_kw)
        assert off_mode.cache_write_in_norm_rope is False and isinstance(off_mode._cache_write, _CacheWrite)
        assert [st.name for st in off_mode._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "sigmoid_gate", "quantize_o", "out_proj"]
    # fp32 activations on the dense DECODE form (declared here, declined by check_support): the activation row
    with pytest.raises(NotImplementedError, match="serves bf16 / f16 activations"):
        _make_block(12, 1, 1, dtype=torch.float32, fuse_cache_write_small_m=True)
    f32 = _make_block(12, 1, 1, dtype=torch.float32)[0]
    assert f32.decode_route is not None and f32.cache_write_in_norm_rope is False and isinstance(f32._cache_write, _CacheWrite)
    # the guard rows, on the resolver directly (a dense decode declaration with the facts forced): quant, the indexer band, the e4m3 cache
    blk = _make_block(12, 1, 1)[0]
    assert blk.cache_write_in_norm_rope is True
    with pytest.raises(NotImplementedError, match="serves the bf16 / f16 pipeline only, got quant=QuantSpec"):
        blk._resolve_cache_write_small_m(True, spec)
    assert blk._resolve_cache_write_small_m(None, spec) is False
    blk.geom = _qsa._geom(_qsa.GEOM_SMALL, index_band=True)[0]
    assert blk.geom.index_band
    with pytest.raises(NotImplementedError, match="indexer band"):
        blk._resolve_cache_write_small_m(True, None)
    assert blk._resolve_cache_write_small_m(None, None) is False
    blk.geom = geom
    blk.kv_cache_e4m3 = True
    with pytest.raises(NotImplementedError, match="QUANTIZES"):
        blk._resolve_cache_write_small_m(True, None)
    assert blk._resolve_cache_write_small_m(None, None) is False


def test_an_explicit_fuse_cache_write_false_keeps_the_separate_launch_under_the_auto_small_m_fold():
    """``fuse_cache_write=False`` is "the separate launch always (the A/B control and the fallback)": under ``fuse_cache_write_small_m=None``
    (AUTO) the request outranks the measured small-M rule -- the norm + RoPE launch carries no page size, the separate cache-write stage is
    built and listed -- at every row count the rule would fold (``S_q = 1`` at any batch, ``B x S_q < 128``), on the C0-lowered
    ``fuse_norm_rope=True`` declaration and on ``fuse_norm_rope=False`` alike; the control's stage list is the AUTO fold's plus exactly
    ``cache_write``, its carve the same.  An explicit ``fuse_cache_write_small_m`` is this arm's own knob and the strongest request: ``True``
    beside ``fuse_cache_write=False`` folds the write-through in (an explicit small-M value wins on either side, as documented), ``False`` keeps
    the separate launch.  ``fuse_cache_write=None`` leaves the AUTO fold as before and ``fuse_cache_write=True`` keeps the fused fork, which
    owns its own write-through (never this arm)."""
    at = lambda b=1, s_q=1, **kw: _make_block(12, b, s_q, **kw)[0]  # noqa: E731
    names = lambda blk: [st.name for st in blk._stages]  # noqa: E731
    # the reviewed cell: a small-M decode declaration (24 / 2 at B 1 x S_q 1) with the control asked for by name
    ctl = at(fuse_cache_write=False)
    assert ctl.fuse_cache_write is False and ctl.fuse_cache_write_small_m is None and ctl.small_m_decode and not ctl.fuse_norm_rope_effective
    assert ctl.cache_write_in_norm_rope is False and isinstance(ctl._cache_write, _CacheWrite) and not ctl.cache_write_fused
    assert ctl._norm_rope is not None and ctl._norm_rope.page_size == 0 and "cache_write" in names(ctl)
    auto = at()
    assert auto.fuse_cache_write is None and auto.cache_write_in_norm_rope is True and auto._cache_write is None
    assert names(auto) == [n for n in names(ctl) if n != "cache_write"], "the control adds exactly the cache-write stage"
    assert ctl._layout() == auto._layout(), "the carve does not read the control (performance-only)"
    # every cell the AUTO rule folds, with the C0 lowering (fuse_norm_rope=True to 128 rows) and on the chain declared unfused outright
    for b, s_q in ((1, 1), (12, 1), (4, 4), (32, 1), (128, 1), (31, 4), (63, 2)):
        for fnr in (True, False):
            blk = at(b, s_q, fuse_norm_rope=fnr, fuse_cache_write=False)
            assert blk._norm_rope is not None and not blk.fuse_norm_rope_effective, (b, s_q, fnr)
            assert blk.cache_write_in_norm_rope is False and isinstance(blk._cache_write, _CacheWrite) and blk._norm_rope.page_size == 0, (b, s_q, fnr)
            assert at(b, s_q, fuse_norm_rope=fnr).cache_write_in_norm_rope is True, (b, s_q, fnr)  # the AUTO twin folds
    # the explicit small-M knob is the strongest request: True beside the control folds in, False keeps the separate launch
    on = at(fuse_cache_write=False, fuse_cache_write_small_m=True)
    assert on.cache_write_in_norm_rope is True and on._cache_write is None and on._norm_rope.page_size == 16 and names(on) == names(auto)
    off = at(fuse_cache_write=False, fuse_cache_write_small_m=False)
    assert off.cache_write_in_norm_rope is False and isinstance(off._cache_write, _CacheWrite) and names(off) == names(ctl)
    # the other readings of fuse_cache_write: None folds as before; True keeps the fused fork (its own write-through, never this arm)
    assert at(fuse_cache_write=None).cache_write_in_norm_rope is True
    fork = at(fuse_norm_rope=True, fuse_cache_write=True)
    assert fork.fuse_norm_rope_effective and fork.cache_write_fused and fork.cache_write_in_norm_rope is False and fork._norm_rope is None
    # past the rule the control changes nothing: 128 rows with S_q > 1 keep the separate launch under every reading
    assert at(32, 4, fuse_norm_rope=False).cache_write_in_norm_rope is False
    assert at(32, 4, fuse_norm_rope=False, fuse_cache_write=False).cache_write_in_norm_rope is False
    # the resolver on its own facts: AUTO reads False with the control declared and True without it; an explicit value is followed either way
    assert ctl._resolve_cache_write_small_m(None, None) is False and auto._resolve_cache_write_small_m(None, None) is True
    assert ctl._resolve_cache_write_small_m(True, None) is True and ctl._resolve_cache_write_small_m(False, None) is False


# ---------------------------------------------------------------------------
# cc 10.0 (the SM100 line of DENSE_SERVING_CCS; 2026-10-29): the host rows on any device, the accept / reject cells on the part
# ---------------------------------------------------------------------------
#
# The block's dense bf16 / f16 serving forms are admitted on cc 10.0 through the shared SM100 d256 kernels: the SM100 decode tile
# (sm100/decode_d256_f16.py) under the line's own route (_SM100_ROUTE_TABLE), the paged prefill tile packed, the split combine in the
# combine module's own forms, the sigmoid gate as its own launch (fuse_gate=True is a typed decline: the sm100 row claims no gate
# epilogue), PDL off by default.  Functional cells only -- nothing measured there; every Rubin-only knob a typed decline by name.


def _sm100_reading(monkeypatch):
    """The cc 10.0 reading of the block's DEVICE on any host: the constructor's route / split / PDL reads and the part rows read this
    one function (the stages' own gates read the real device: a host row that calls ``check_support`` reaches the part rows first)."""
    monkeypatch.setattr(_api, "_device_cc_of", lambda device: _SM100)


def test_on_a_cc_10_0_reading_the_declaration_records_the_sm100_route(monkeypatch):
    """A DECODE block declared on a cc 10.0 reading records the SM100 line's route (the adapter's rule, the table above) with the
    declared batch and KV bound, the split from the Blackwell fit (no floor: the gate is never in the combine there), the sigmoid gate
    as its own stage, PDL AUTO False with the explicit switch honoured, the prefill route's THD binding at the dense envelope; the
    declaration-time declines of the line -- the cap, an unpacked step past 16 tokens off the wide rule -- raised with the route's reason."""
    _sm100_reading(monkeypatch)
    for group in sorted(_GROUPS):
        for s_q in (1, 2, 3, 4, 5, 9, 16):
            want = _SM100_ROUTE_TABLE[group][s_q]
            blk, _ = _make_block(group, 4, s_q)
            r = blk.decode_route
            assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == want and (r.device_cc, r.batch, r.max_kv_len) == (_SM100, 4, 64 * 16), (
                group,
                s_q,
                r,
            )
            assert r == decode_route(blk.geom, s_q, batch=4, device_cc=_SM100, max_kv_len=64 * 16) and not r.claimed
            assert (
                blk.use_pdl is False and blk.use_pdl_requested is None and blk.pdl_stages == () and not blk.gate_fused and isinstance(blk._gate, _SigmoidGate)
            )
            assert blk._sdpa.decode_route == r and blk.decode_split_kv >= 1 and blk._sdpa.decode_split_kv == blk.decode_split_kv
            if r.prefill_tile:
                assert blk._sdpa.thd and blk.decode_split_kv == 1 and (blk._sdpa.num_sequences, blk._sdpa.max_seq_len) == (4, s_q)
            else:
                assert blk._sdpa.paged_dense and not blk._sdpa.thd and blk._sdpa.seq_lens_present
            names = [st.name for st in blk._stages]
            assert names == ["qkv_gate_proj", "qk_norm_rope", "sdpa", "sigmoid_gate", "out_proj"], names  # fuse_norm_rope=False: the unfused chain
            on, _ = _make_block(group, 4, s_q, use_pdl=True)
            assert on.use_pdl is True and ("sdpa" in on.pdl_stages) is r.decode_tile and on._sdpa.use_pdl is r.decode_tile
        # fuse_norm_rope=True at decode row counts LOWERS to the unfused chain on this part too (the fork would be the sm_107a decline)
        low, _ = _make_block(group, 4, 1, fuse_norm_rope=True)
        assert low.fuse_norm_rope and not low.fuse_norm_rope_effective and low.small_m_decode and low._norm_rope is not None and low.cache_write_in_norm_rope
        # the declaration-time declines of the line: the cap (the Rubin table's knob) and the unpacked 17 .. 32-token step off the wide rule
        with pytest.raises(NotImplementedError, match="decode_q_tile_max=32 caps the Rubin"):
            _make_block(group, 4, 1, decode_q_tile_max=32)
        with pytest.raises(NotImplementedError, match="decode_q_tile_max=16 caps the Rubin"):
            _make_block(group, 4, 1, decode_q_tile_max=16)
        if group != 4:
            with pytest.raises(NotImplementedError, match="WIDE launches only") as e:
                _make_block(group, 4, 17)
            assert "32-column" in str(e.value) and f"batch 4 x H_q {_GROUPS[group][0]}" in str(e.value) and "EXTEND form" in str(e.value)
            assert _WIDE_ARM_LIFT in str(e.value)  # the same arm, the lift named
        env = GatedAttentionBlockFwd.serving_envelope(_geom(group)[0], torch.bfloat16, _DEV, device_cc=_SM100)
        assert env.decode.route(17, batch=4, max_kv_len=64 * 16).served is (group == 4), "the envelope predicts the declaration's verdict"
    # the wide launch through the declaration: group 6 at 32 sequences over a 4096-key bound is the adapter's 32-column arm packed at S_q 3
    # -- a typed decline naming the lift until the arm's cells run (DECODE_WIDE_TILE_CCS), predicted by the envelope at the same inputs
    with pytest.raises(NotImplementedError, match="the 32-column SM100 decode arm: its cc 10.0 accept cells have not run") as e:
        _make_block(6, 32, 3, max_kv_len=4096, n_pages=256)
    assert "at a WIDE launch" in str(e.value) and "batch 32 x H_kv 4 = 128 units over max_kv_len=4096" in str(e.value)
    env6 = GatedAttentionBlockFwd.serving_envelope(_geom(6)[0], torch.bfloat16, _DEV, device_cc=_SM100).decode
    assert not env6.route(3, batch=32, max_kv_len=4096).served and env6.route(3, batch=32, max_kv_len=4096).reason in str(e.value)
    with monkeypatch.context() as m:  # the one-tuple lift through the declaration: the same inputs record the 32-column tile packed
        m.setattr(_api, "DECODE_WIDE_TILE_CCS", (_SM100,))
        wide, _ = _make_block(6, 32, 3, max_kv_len=4096, n_pages=256)
        assert (wide.decode_route.n_q, wide.decode_route.pack_gqa, wide.decode_route.max_kv_len, wide.decode_route.batch) == (32, True, 4096, 32)
    # the Rubin pins of the cell matrix hold on the OTHER reading of the same host (no cross-talk between the two readings)
    monkeypatch.setattr(_api, "_device_cc_of", lambda device: _SM107)
    blk, _ = _make_block(16, 4, 2)
    assert blk.decode_route.n_q == 32 and blk.decode_route.pack_gqa and blk.decode_route.device_cc == _SM107 and blk.use_pdl is True


def test_on_a_cc_10_0_reading_the_rubin_only_pipelines_are_declined_by_name_before_any_stage(monkeypatch):
    """``check_support`` on a cc 10.0 reading declines the Rubin-only pipelines BEFORE any stage runs, each message naming the lift:
    ``quant`` (the FP8 / MXFP8 d256 SDPA rows + the sm_107a forks), ``save_for_backward`` (the Rubin SDPA backward), ``thd=True`` over
    the block's own K / V (no packed self-attention cell through the block there), and a QsaSpec geometry on any kv_source (the sparse
    core, the lean tile through the block, the indexer, the e4m3 cache -- the sparse part statement, raised by the block before the
    indexer stages and by the sparse stage itself).  A host row: the declaration is device-free and the part rows precede every stage."""
    from cudnn.gated_attention_block import QuantSpec

    _sm100_reading(monkeypatch)
    geom, _ = _geom(8)  # 8 / 1 at d_model 512
    mk = lambda *shape, dt=torch.bfloat16: torch.empty(*shape, dtype=dt, device=_DEV)  # noqa: E731
    dense = (
        mk(1, 256, _D_MODEL),
        mk(geom.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(1, 256, _ROPE),
        mk(1, 256, _ROPE),
        mk(_D_MODEL, 8 * _D),
        mk(1, 256, _D_MODEL),
        geom,
    )
    with pytest.raises(NotImplementedError, match="save_for_backward=True on cc 10.0") as e:
        GatedAttentionBlockFwd(*dense, save_for_backward=True, inplace_qkv=False).check_support()
    assert "SdpaBwdDslSm107" in str(e.value) and "the lift" in str(e.value)
    e4 = lambda *shape: torch.zeros(*shape, dtype=torch.float8_e4m3fn, device=_DEV)  # noqa: E731
    spec = QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)
    fp8 = (
        e4(1, 256, _D_MODEL),
        e4(geom.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(1, 256, _ROPE),
        mk(1, 256, _ROPE),
        e4(_D_MODEL, 8 * _D),
        mk(1, 256, _D_MODEL),
        geom,
    )
    with pytest.raises(NotImplementedError, match="quant=QuantSpec on cc 10.0") as e:
        GatedAttentionBlockFwd(*fp8, quant=spec).check_support()
    assert "FP8-projection + bf16-attention" in str(e.value) and "FP8_PROJ_BF16_ATTENTION_CCS" in str(e.value)
    # the FP8-projection + bf16-attention mode on the cc 10.0 reading: the part statement of its OWN tuple -- declined by name while
    # FP8_PROJ_BF16_ATTENTION_CCS lacks the part, the part rows passed once it holds it (the one-tuple lift)
    mode = QuantSpec.bf16_attention(0.01, 0.02, 0.03, 4.0)
    kc, vc = _pools(8, 1, 16, _D, "hnd", torch.bfloat16)
    mode_args = (
        e4(4, 1, _D_MODEL),
        e4(geom.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(4, 1, _ROPE),
        mk(4, 1, _ROPE),
        e4(_D_MODEL, 8 * _D),
        mk(4, 1, _D_MODEL),
        geom,
    )
    mode_kw = dict(
        paged_kv_page_size=16, kv_source="paged", sample_k_cache=kc, sample_v_cache=vc, sample_block_table=torch.zeros(4, 8, dtype=torch.int32, device=_DEV)
    )
    mode_blk = GatedAttentionBlockFwd(*mode_args, quant=mode, fuse_gate=False, **mode_kw)
    assert mode_blk.attention_bf16 and mode_blk.decode_route.device_cc == _SM100
    mode_blk._check_declaration()
    if _SM100 in FP8_PROJ_BF16_ATTENTION_CCS:
        mode_blk._check_part_support()
    else:
        with pytest.raises(NotImplementedError, match=r"QuantSpec\(attention_dtype=torch.bfloat16\) on cc 10.0") as e:
            mode_blk._check_part_support()
        assert "FP8_PROJ_BF16_ATTENTION_CCS" in str(e.value) and "the lift" in str(e.value)
    with monkeypatch.context() as m:
        m.setattr(_api, "FP8_PROJ_BF16_ATTENTION_CCS", (_SM107, _SM100))
        mode_blk._check_part_support()
    t = 256
    packed = (mk(t, _D_MODEL), mk(geom.n_qkvg, _D_MODEL), mk(_D), mk(_D), mk(t, _ROPE), mk(t, _ROPE), mk(_D_MODEL, 8 * _D), mk(t, _D_MODEL), geom)
    with pytest.raises(NotImplementedError, match="thd=True over the block's own K / V") as e:
        GatedAttentionBlockFwd(*packed, thd=True, num_sequences=2, max_seq_len=128).check_support()
    assert "the EXTEND form" in str(e.value) and "the lift" in str(e.value)
    # the EXTEND form itself (thd=True with kv_source='paged') and the dense inference form pass the part rows (the stages speak next:
    # on this sm80 host the first stage's own decline, never the part statement)
    ok = GatedAttentionBlockFwd(*dense)
    ok._check_declaration()
    ok._check_part_support()
    # a QsaSpec geometry: the sparse part statement at the block level and on the sparse stage itself
    sg = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True, qsa=QsaSpec())
    kc, vc = _pools(8, 2, 16, _D, "hnd", torch.bfloat16)
    sblk = GatedAttentionBlockFwd(
        mk(1, 1, _D_MODEL),
        mk(sg.n_qkvg, _D_MODEL),
        mk(_D),
        mk(_D),
        mk(1, 1, _ROPE),
        mk(1, 1, _ROPE),
        mk(_D_MODEL, 8 * _D),
        mk(1, 1, _D_MODEL),
        sg,
        paged_kv_page_size=16,
        kv_source="paged",
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(1, 8, dtype=torch.int32, device=_DEV),
        sparse_tile_form="lean",
    )
    assert sblk.sparse_tile_form == "lean", "the declaration resolves the explicit lean body (the adapter admits it standalone on cc 10.0)"
    with pytest.raises(NotImplementedError, match=r"sparse forms .* served on cc 10\.7 \(Rubin\) only") as e:
        sblk.check_support()
    assert "explicit lean tile on cc 10.0 / 10.3" in str(e.value) and "the lift" in str(e.value)
    with pytest.raises(NotImplementedError, match=r"sparse forms .* served on cc 10\.7 \(Rubin\) only"):
        sblk._sdpa.check_support()  # the sparse stage's own gate, before the adapter


def _sm100_case(key, *args, **kw):
    """One compiled cc 10.0 DECODE cell per key: ``fuse_gate=False`` (the served gate placement there) unless the cell says otherwise."""
    blk_kw = dict(fuse_gate=False)
    blk_kw.update(kw.pop("blk_kw", None) or {})
    return _case(("sm100",) + tuple(key), *args, blk_kw=blk_kw, **kw)


def _sm100_pin(case, want):
    """The compiled kernel against the declared SM100 route: the SM100 decode tile (its file, the width, the packing) or the shared
    paged prefill tile with the group packed; no gate epilogue anywhere, the gate its own launch; block 0 untouched."""
    r, rb = case.blk.decode_route, case.route_readback()
    assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == want and r.device_cc == _SM100 and not r.claimed, (r, want)
    assert rb["template"] == r.tile and rb["adapter_split"] == case.blk.decode_split_kv and (rb["adapter_pack"] is True) == r.pack_gqa, (rb, r)
    assert not case.blk.gate_fused and isinstance(case.blk._gate, _SigmoidGate) and not case.blk._sdpa.fuse_gate
    if r.decode_tile:
        assert rb["file"].endswith("/sm100/decode_d256_f16.py") and rb["heads_per_tile"] == (r.group if r.pack_gqa else 1), rb
        assert rb["n_q"] == r.n_q and rb["split"] == case.blk.decode_split_kv and rb["epilogue_gate"] is False, rb
    else:
        assert r.prefill_tile and rb["file"].endswith("/sm100/prefill_d256_f16.py") and rb["heads_per_tile"] == r.group and rb["n_q"] is None, rb
        assert bool(rb["epilogue_gate"]) is False and case.blk.decode_split_kv == 1, rb
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0), "block 0 (the NULL block) must never be written"
    return r, rb


@requires_blackwell
@pytest.mark.parametrize("group", sorted(_GROUPS))
@pytest.mark.parametrize("s_q", [1, 2, 3, 4, 5])
def test_sm100_decode_matches_the_oracle_at_every_group_and_s_q(group, s_q):
    """The cc 10.0 DECODE cells at every served group x S_q 1 .. 5 against the per-sequence fp32 oracle under the dense suite's budget
    (cos > 0.999 on ``out``; nothing widened), the padded entry EXACTLY 0 with block 0 untouched, the compiled kernel pinned against the
    SM100 line's route: the 16-column SM100 decode tile packed (S_q 1 everywhere, 2 at groups 8 / 6, 1 .. 4 at group 4), UNPACKED with
    ``group`` KV streams at groups 6 / 12 past it, the paged prefill tile packed at groups 16 / 8 / 4; the gate its own launch."""
    case = _sm100_case(("std", group, s_q), group, s_q, _LENS)
    case.run()
    case.check_against_oracle()
    _sm100_pin(case, _SM100_ROUTE_TABLE[group][s_q])


@requires_blackwell
@pytest.mark.parametrize("s_q, page", [(1, 16), (4, 32)], ids=["sq1-p16", "sq4-p32"])
def test_sm100_decode_the_27b_geometry_matches_the_oracle(s_q, page):
    """The 24 / 4 geometry at its real width (``d_model`` 5120, n_qkvg 14336: a multiple of 256) on cc 10.0: S_q 1 at page 16 on the
    16-column tile packed, S_q 4 at page 32 on the 16-column tile UNPACKED (24 rows exceed the small launch's width: 6 KV streams)."""
    case = _sm100_case(("q38_27b", s_q, page), 6, s_q, _LENS, page=page, heads=(24, 4), d_model=5120)
    case.run()
    case.check_against_oracle(what=f"27B 24/4 d_model 5120 S_q {s_q} p{page}")
    _sm100_pin(case, _SM100_ROUTE_TABLE[6][s_q])
    assert case.geom.d_model == 5120 and case.geom.n_qkvg % 256 == 0


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("page", SERVING_PAGE_SIZES)
def test_sm100_l1_l2_l3_pools_are_bitwise_at_every_page_size(page, dtype):
    """The pool-layout matrix on cc 10.0 (the tracker's cc 10.0 matrix): the three stride patterns holding the same values give a
    BITWISE ``out`` at pages 16 / 32 / 64 / 128, bf16 and f16, at 24 / 2 S_q 2 (the 16-column tile unpacked, 12 streams), each
    within the oracle's budget."""
    outs = {}
    for layout in _LAYOUTS:
        case = _sm100_case(("layout", page, layout, dtype), 12, 2, (200, 0, 77, 1000), page=page, layout=layout, dtype=dtype, seed=3)
        outs[layout] = case.run().clone()
        case.check_against_oracle(what=f"cc 10.0 {layout} p{page} {dtype}")
        _sm100_pin(case, _SM100_ROUTE_TABLE[12][2])
    for layout in _LAYOUTS[1:]:
        assert torch.equal(outs[layout], outs["hnd"]), f"{layout} != hnd at page {page} {dtype}"


@requires_blackwell
def test_sm100_two_launches_are_bitwise_and_the_workspace_is_honest():
    case = _sm100_case(("two", 6, 2), 6, 2, _LENS)
    a = case.run().clone()
    b = case.run().clone()
    assert torch.equal(a, b)
    need = case.blk.get_workspace_size()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    out = torch.empty_like(case.out)
    case.execute(out=out, ws=ws)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all() and torch.equal(out, a)


@requires_blackwell
def test_sm100_capture_of_the_first_execute_replays_bitwise_and_takes_new_lengths_tables_slots_and_inputs():
    """The capture / replay cell on cc 10.0 (the serving stack's defaults: a fresh declaration, zero warm-ups, no host sync): the
    default chain there -- the unfused chain with the write-through folded into the norm + RoPE launch, the plain (no PDL) decode tile
    and its combine, the separate gate."""
    _capture_replay_cell(fuse_gate=False)


@requires_blackwell
def test_sm100_the_explicit_pdl_switch_is_served_and_bitwise_the_plain_chain():
    """``use_pdl=True`` is served on cc 10.0 (the SM100 decode tile and the combine carry the griddepcontrol pair; AUTO reads False: the
    default is the measured part's): the switched chain BITWISE the plain one on ``out`` and on the pools, the stage list the same."""
    off = _sm100_case(("pdl", 16, 1, False), 16, 1, _LENS, blk_kw=dict(use_pdl=False))
    on = _sm100_case(("pdl", 16, 1, True), 16, 1, _LENS, blk_kw=dict(use_pdl=True))
    assert on.blk.use_pdl is True and off.blk.use_pdl is False and "sdpa" in on.blk.pdl_stages and on.blk._sdpa.use_pdl is True
    assert [st.name for st in on.blk._stages] == [st.name for st in off.blk._stages]
    a = off.run().clone()
    b = on.run().clone()
    on.check_against_oracle(b, what="cc 10.0 use_pdl=True")
    assert torch.equal(a, b) and torch.equal(_bits(on.k_cache), _bits(off.k_cache)) and torch.equal(_bits(on.v_cache), _bits(off.v_cache))
    auto = _sm100_case(("pdl", 16, 1, None), 16, 1, _LENS)
    assert auto.blk.use_pdl is False and auto.blk.use_pdl_requested is None, "AUTO reads False off the measured part"


@requires_blackwell
def test_sm100_declines_the_rubin_only_knobs_by_name():
    """On the part itself: ``fuse_gate=True`` is the SM100 row's typed decline at ``check_support`` (the three dense forms), the fused
    projection fork (``fuse_norm_rope_small_m=True``, an explicit ``fuse_cache_write=True``) the sm_107a rendering's, the width cap, the
    unpacked 17-token step and the 32-column arm at a wide launch (``DECODE_WIDE_TILE_CCS``) the declaration's; the served twins build
    and run (the oracle cells above)."""
    gated, _ = _make_block(16, 4, 1, fuse_gate=True)
    with pytest.raises(NotImplementedError, match="fuse_gate=True is not served by the f16/bf16 SM100 SDPA engine row") as e:
        gated.check_support()
    assert "cc 10.0" in str(e.value) and "fuse_gate=False" in str(e.value)
    for kw in (dict(fuse_norm_rope=True, fuse_norm_rope_small_m=True), dict(fuse_norm_rope=True, fuse_cache_write=True)):
        fork, _ = _make_block(16, 4, 1, **kw)
        assert fork.fuse_norm_rope_effective and fork._norm_rope is None
        with pytest.raises(NotImplementedError, match=r"rendered for sm_107a \(Rubin\); this device is SM100"):
            fork.check_support()
    with pytest.raises(NotImplementedError, match="decode_q_tile_max=32 caps the Rubin"):
        _make_block(16, 4, 1, decode_q_tile_max=32)
    with pytest.raises(NotImplementedError, match="WIDE launches only"):
        _make_block(6, 4, 17)
    with pytest.raises(NotImplementedError, match="the 32-column SM100 decode arm: its cc 10.0 accept cells have not run"):
        _make_block(6, 32, 3, max_kv_len=4096, n_pages=256)  # the adapter's 32-column arm at a wide launch: declined by name on the part
    assert _cc() == _SM100 and GatedAttentionBlockFwd.serving_envelope(_geom(16)[0], torch.bfloat16, _DEV).forms == ("extend", "decode")
    # the FP8-projection + bf16-attention mode ON the part: served iff the tuple holds cc 10.0 (its cells below run then), else the part
    # statement at check_support and no spelling in the envelope's quant_modes
    mode = _mode_declaration(16, 4, 1)
    env = GatedAttentionBlockFwd.serving_envelope(_geom(16)[0], torch.float8_e4m3fn, _DEV, quant=mode.quant)
    if _SM100 in FP8_PROJ_BF16_ATTENTION_CCS:
        mode._check_declaration()
        mode._check_part_support()
        assert env.forms == ("extend", "decode") and env.decode.quant_modes == (QUANT_MODE_FP8_PROJ_BF16_ATTENTION,)
    else:
        with pytest.raises(NotImplementedError, match=r"QuantSpec\(attention_dtype=torch.bfloat16\) on cc 10.0") as e:
            mode.check_support()
        assert "FP8_PROJ_BF16_ATTENTION_CCS" in str(e.value) and env.forms == () and env.decode.quant_modes == () and env.decode.reason == str(e.value)


# ---------------------------------------------------------------------------
# Both served parts (requires_dense_part; 2026-10-29): a serving stack's PACKED K|V cache rows bound as two half views
# ---------------------------------------------------------------------------
#
# A serving stack keeps ONE cache tensor per layer, [num_pages, H_kv, P, 2D] -- K in columns [0, D) and V in [D, 2D) of every row (the HND cache
# of the 24 / 4 model family: [pages, 4, 16 or 32, 512] bf16) -- and hands the block its two halves as the pools, kv[..., :D] / kv[..., D:]: the
# "unified" layout of the matrix above at the stack's OWN shape.  The cells here mirror that binding literally -- 24 / 4 at d 256, page 16 with
# one token per sequence and page 32 with a four-token speculative step, a 2048-key context next to a padded entry and a short one, the stack's
# int64 slot mapping -- and prove the ONE tensor behind the views: the step's rows land in the K half and the V half alone (bitwise the slab
# bands attention consumed), the prefix rows are untouched, and every byte of every row nobody addressed -- block 0 among them -- still holds
# the sentinel in BOTH halves (a K store running past D would land in the V half of the same row, and vice versa); the oracle, two launches
# bitwise, bitwise the HND-compact twin.  Both parts: the gate in the combine on cc 10.7, its own launch on cc 10.0.

_PACKED_KV_LENS = (2048, 0, 5, 1000, 33)  # the stack's longest context, a PADDED entry, a short one (kv_lens >= S_q at S_q 4), two tails
_PACKED_KV_CELLS = [pytest.param(16, 1, id="p16-plain"), pytest.param(32, 4, id="p32-mtp4")]  # (page, S_q): one token per sequence / a 4-token step


def _packed_kv_case(page, s_q, dtype, *, layout="unified", seed=11):
    """ONE compiled 24 / 4 DECODE cell per (page, S_q, dtype, layout) on the served part's default chain: the gate in the split combine on cc
    10.7, its own launch on cc 10.0 (``fuse_gate=False``, the placement the SM100 row serves)."""
    blk_kw = {} if _cc() == _SM107 else dict(fuse_gate=False)
    return _case(("packed_kv", page, s_q, dtype, layout), 6, s_q, _PACKED_KV_LENS, page=page, layout=layout, dtype=dtype, seed=seed, blk_kw=blk_kw)


def _run_with_slots(case, slot_dtype):
    """One execute with the slot mapping in ``slot_dtype`` (the stack's int64), synchronized; ``out`` cloned."""
    case.execute(slot_mapping=case.slot_mapping.to(slot_dtype))
    torch.cuda.synchronize()
    return case.out.clone()


def _assert_packed_kv_halves(case, what):
    """The ONE tensor behind ``case.k_cache`` / ``case.v_cache`` and the row contract on it after an execute: the two views are the halves of a
    contiguous ``[num_pages, H_kv, P, 2D]`` tensor (the V view's base ``D`` elements past the K view's, both with strides ``(H_kv*P*2D, P*2D,
    2D, 1)``); the step's slots hold the slab's K band in the K half and the slab's V band in the V half, bitwise; the prefix rows (written by
    torch before the run) are unchanged in both halves; every row nobody addressed -- the NULL block 0, the stale rows past ``kv_lens`` on a
    sequence's last page, the spare pages -- still holds the sentinel in BOTH halves.  Returns the number of addressed rows (slots)."""
    kv, page, D = case.k_cache._base, case.page, _D
    h_kv = int(case.k_cache.shape[1])
    assert kv is case.v_cache._base and tuple(kv.shape) == (case.n_pages, h_kv, page, 2 * D) and kv.is_contiguous(), f"{what}: not ONE [P, H, page, 2D] tensor"
    assert (
        case.k_cache.data_ptr() == kv.data_ptr() and case.v_cache.data_ptr() == kv.data_ptr() + D * kv.element_size()
    ), f"{what}: the views are not the two halves"
    assert tuple(case.k_cache.stride()) == tuple(case.v_cache.stride()) == (h_kv * page * 2 * D, page * 2 * D, 2 * D, 1), f"{what}: the half views' strides"
    _, _, k_band, v_band = _slab_bands(case)
    slot = case.slot_mapping
    live = slot >= 0
    s = slot[live].long()
    assert torch.equal(_bits(case.k_cache[s // page, :, s % page, :]), _bits(k_band[live])), f"{what}: the K half at the step's slots is not the slab's K band"
    assert torch.equal(_bits(case.v_cache[s // page, :, s % page, :]), _bits(v_band[live])), f"{what}: the V half at the step's slots is not the slab's V band"
    addressed = torch.zeros(case.n_pages * page, dtype=torch.bool, device=_DEV)
    for b, L in enumerate(case.kv_lens):
        if L == 0:
            continue
        pos = torch.arange(L, device=_DEV)
        slots_b = case.block_table[b, : -(-L // page)].long()[pos // page] * page + pos % page
        addressed[slots_b] = True
        n_pre = L - case.s_q
        if n_pre:
            sp = slots_b[:n_pre]
            k_ref, v_ref = (x[:n_pre].to(case.dtype) for x in case.ref_rows[b])
            assert torch.equal(_bits(case.k_cache[sp // page, :, sp % page, :]), _bits(k_ref)), f"{what}: a prefix K row of sequence {b} changed"
            assert torch.equal(_bits(case.v_cache[sp // page, :, sp % page, :]), _bits(v_ref)), f"{what}: a prefix V row of sequence {b} changed"
    assert not bool(addressed[:page].any()), f"{what}: block 0 is the NULL block, never addressed"
    rows = kv.permute(0, 2, 1, 3).reshape(case.n_pages * page, h_kv, 2 * D)  # one [H_kv, 2D] content row per slot
    assert bool((rows[~addressed] == _SENTINEL).all()), f"{what}: a row nobody addressed was written, in one of the two halves"
    return int(addressed.sum().item())


@requires_dense_part
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("page, s_q", _PACKED_KV_CELLS)
def test_packed_kv_rows_of_a_serving_stack_bind_as_two_half_views_on_the_decode_form(page, s_q, dtype):
    """The stack's ONE ``[pages, 4, P, 512]`` cache tensor bound as the two half views on the DECODE form at 24 / 4: page 16 with one token per
    sequence, page 32 with a four-token speculative step, kv 2048 / 0 (padded) / 5 / 1000 / 33, int64 slots, bf16 and f16 -- ``out`` within the
    oracle's budget (the padded entry exactly 0), the write-through in the right halves and nothing else touched (``_assert_packed_kv_halves``),
    two launches bitwise, bitwise the HND-compact twin on the same inputs, the route pinned against the part's table."""
    case = _packed_kv_case(page, s_q, dtype)
    what = f"packed K|V rows 24/4 p{page} S_q {s_q} {dtype}"
    out = _run_with_slots(case, torch.int64)
    case.check_against_oracle(out, what=what)
    n_rows = _assert_packed_kv_halves(case, what)
    assert (
        n_rows == sum(_PACKED_KV_LENS) and _written(case.k_cache) == _written(case.v_cache) == n_rows * 4
    ), f"{what}: the written rows are not the addressed ones"
    again = _run_with_slots(case, torch.int64)
    assert torch.equal(out, again), f"{what}: two launches differ"
    _assert_packed_kv_halves(case, what + " (second launch)")
    twin = _packed_kv_case(page, s_q, dtype, layout="hnd")
    assert twin.k_cache._base is None and twin.k_cache.is_contiguous(), "the twin: two compact tensors"
    out_h = _run_with_slots(twin, torch.int64)
    twin.check_against_oracle(out_h, what=what + " (HND twin)")
    assert torch.equal(out, out_h), f"{what}: out differs from the HND-compact twin's: max|diff| {(out.float() - out_h.float()).abs().max().item():.3e}"
    r = case.blk.decode_route
    if _cc() == _SM107:
        assert (r.tile, r.n_q, r.token_units, r.pack_gqa) == _ROUTE_TABLES[DECODE_ROUTE_SINGLE_UNIT][6][s_q] and case.blk.gate_fused, r
        assert case.route_readback()["file"].endswith("/sm107/decode_d256_f16.py")
    else:
        _sm100_pin(case, _SM100_ROUTE_TABLE[6][s_q])


# ---------------------------------------------------------------------------
# The FP8-PROJECTION + BF16-ATTENTION mode (QuantSpec(attention_dtype=torch.bfloat16)) on the DECODE form
# ---------------------------------------------------------------------------
#
# A serving stack whose attention projections are static per-tensor FP8 W8A8 while its KV cache and attention math are bf16: e4m3 h / W_qkvg
# / W_o under one scale each, the two FP8 forced-tile projection GEMMs with the descale folded into their bf16 epilogue, QK-norm + RoPE and
# the pool write-through in bf16, stage (4) the bf16 serving form's own kernels and routes over bf16 pools (the gate in the combine on cc 10.7,
# its own launch on cc 10.0), the gated O quantized to e4m3 at scale_o (quantize_o: the mode's ONE quantize launch) for the FP8 out
# projection.  The cells are the bf16 DECODE cells' twins through `_Case(quant=True)`: the oracle is the fp8 reference's attention_bf16 arm
# (no Q / K / V quantization point; the gated O quantized at scale_o exactly as the block does), the bound the SIBLING's (cos > 0.999 on out,
# a padded entry EXACTLY 0, block 0 untouched) -- never the fp8 suite's 0.99 (every quantization point of the mode is replicated).  Served on
# the parts of FP8_PROJ_BF16_ATTENTION_CCS: the cells skip by name off them (a part off the tuple declines the mode at check_support, pinned
# in test_sm100_declines_the_rubin_only_knobs_by_name on cc 10.0 and in the host rows above).

requires_fp8_proj_bf16_attention_part = pytest.mark.skipif(
    _cc() not in FP8_PROJ_BF16_ATTENTION_CCS,
    reason=f"the FP8-projection + bf16-attention mode is served on {FP8_PROJ_BF16_ATTENTION_CCS} (declined by name elsewhere); found {_cc()}",
)


def _mode_part_kw():
    """The mode's declaration knobs per served part: the gate in the combine on cc 10.7 (the `_Case` default, fuse_gate=True), its own launch
    on every other part (the sm100 row claims no gate epilogue)."""
    return {} if _cc() == _SM107 else dict(fuse_gate=False)


def _mode_case(key, *args, blk_kw=None, **kw):
    """ONE compiled mode cell per key (the cache the bf16 cells use), on the part's default chain."""
    bk = _mode_part_kw()
    bk.update(blk_kw or {})
    return _case(("mode",) + tuple(key), *args, quant=True, blk_kw=bk, **kw)


def _mode_declaration(group, batch, s_q, **blk_kw):
    """A declared (not compiled) mode block on EMPTY e4m3 tensors over bf16 pools: the declaration cells."""
    geom, _ = _geom(group)
    h_q, h_kv = _GROUPS[group]
    e4 = lambda *shape: torch.zeros(*shape, dtype=torch.float8_e4m3fn, device=_DEV)  # noqa: E731
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    kc, vc = _pools(64, h_kv, 16, _D, "hnd", torch.bfloat16)
    kw = dict(
        paged_kv_page_size=16,
        kv_source="paged",
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(batch, 64, dtype=torch.int32, device=_DEV),
    )
    kw.update(_mode_part_kw())
    kw.update(blk_kw)
    return GatedAttentionBlockFwd(
        e4(batch, s_q, _D_MODEL),
        e4(geom.n_qkvg, _D_MODEL),
        torch.ones(_D, dtype=torch.bfloat16, device=_DEV),
        torch.ones(_D, dtype=torch.bfloat16, device=_DEV),
        mk(batch, s_q, _ROPE),
        mk(batch, s_q, _ROPE),
        e4(_D_MODEL, h_q * _D),
        mk(batch, s_q, _D_MODEL),
        geom,
        quant=QuantSpec.bf16_attention(0.01, 0.02, 0.03, 4.0),
        **kw,
    )


def _mode_pin(case):
    """The compiled mode block's facts: the attribute, a bf16 SDPA stage reading the slab in place, no quantize_q / quantize_kv, ONE distinct
    quantize_o, both FP8 projections on the jit-only capture-safe route with the alpha epilogue, the carve without the compact e4m3 Q / K / V,
    the route pinned on the compiled module (the part's table: the Rubin readback on cc 10.7, `_sm100_pin` on cc 10.0), block 0 untouched."""
    blk = case.blk
    assert blk.attention_bf16 and blk.quant.attention_bf16 and blk.dtype == torch.float8_e4m3fn and blk.act_dtype == torch.bfloat16 and blk.inplace_qkv
    names = [st.name for st in blk._stages]
    assert (
        names.count("quantize_o") == 1 and "quantize_q" not in names and "quantize_kv" not in names and names[0] == "qkv_gate_proj" and names[-1] == "out_proj"
    )
    assert names.index("quantize_o") == len(names) - 2 and names.index("sdpa") < names.index("quantize_o")
    sd = blk._sdpa
    assert isinstance(sd, _Sdpa) and sd.dtype == torch.bfloat16 and sd.fp8 is False and not sd.mxfp8 and sd.o_dtype == torch.bfloat16
    assert sd.token_stride == blk.geom.n_qkvg and sd.gate_token_stride == blk.geom.n_qkvg and sd.kv_source == KV_SOURCE_PAGED
    assert blk._quant_q is None and blk._quant_kv is None and isinstance(blk._quant_o, _Quantize) and blk._quant_o.heads == blk.geom.h_q
    for st in (blk._proj, blk._out_proj):
        assert isinstance(st, _Projection) and st.dtype == torch.float8_e4m3fn and st.alpha and st.capture_safe and st._plan.route == "jit-only", (
            st.name,
            st._plan.route,
        )
    lay = blk._ws
    assert lay.q8 == lay.k8 == lay.v8 == -1 and lay.o8 >= 0 and lay.q == lay.k == lay.v == -1
    assert set(blk._quant_dev) == {"alpha_qkvg", "alpha_o", "scale_o"} and blk.use_pdl is False
    r, rb = blk.decode_route, case.route_readback()
    assert rb["template"] == r.tile and rb["adapter_split"] == blk.decode_split_kv and (rb["adapter_pack"] is True) == r.pack_gqa, (rb, r)
    if _cc() == _SM107:
        assert r.device_cc == _SM107 and (blk.gate_fused == (r.prefill_tile or blk.decode_split_kv >= 2))
        if r.decode_tile:
            assert rb["file"].endswith("/sm107/decode_d256_f16.py") and rb["n_q"] == r.n_q
    else:
        _sm100_pin(case, _SM100_ROUTE_TABLE[r.group][r.s_q])
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0), "block 0 (the NULL block) must never be written"
    return r, rb


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
@pytest.mark.parametrize("group", sorted(_GROUPS))
@pytest.mark.parametrize("s_q", [1, 2, 3, 4, 5])
def test_fp8_proj_bf16_attention_decode_matches_the_oracle_at_every_group_and_s_q(group, s_q):
    """The mode's DECODE cells of the served-group matrix: every served group x S_q 1 .. 5 at the standard lengths (a tail tile, the
    ring depth, four tiles with a tail, a PADDED entry) against the mode's oracle under the SIBLING's budget (cos > 0.999 on ``out``,
    the padded entry EXACTLY 0, block 0 untouched), the route pinned against the compiled module as the bf16 cell of the same
    declaration (the part's table), the stage list the bf16 chain's plus the one quantize launch, no compact e4m3 Q / K / V."""
    case = _mode_case(("std", group, s_q), group, s_q, _LENS)
    case.run()
    case.check_against_oracle()
    _mode_pin(case)


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
@pytest.mark.parametrize("page, s_q", _PACKED_KV_CELLS)
def test_fp8_proj_bf16_attention_decode_the_27b_geometry_on_packed_kv_rows(page, s_q):
    """The 24 / 4 geometry at its real width (``d_model`` 5120, n_qkvg 14336) under the mode on a serving stack's PACKED K|V cache (the two
    half views of one ``[pages, 4, P, 512]`` tensor, int64 slots): page 16 x S_q 1 and page 32 x S_q 4 over kv 2048 / 0 (padded) / 5 / 1000 /
    33 -- the oracle, the write-through in the right halves with nothing else touched, two launches bitwise, the route pinned per part."""
    case = _mode_case(("q38_27b_packed", page, s_q), 6, s_q, _PACKED_KV_LENS, page=page, layout="unified", heads=(24, 4), d_model=5120, seed=11)
    what = f"mode 24/4 d_model 5120 packed K|V p{page} S_q {s_q}"
    assert case.geom.d_model == 5120 and case.geom.n_qkvg == 14336 and case.h.dtype == torch.float8_e4m3fn
    out = _run_with_slots(case, torch.int64)
    case.check_against_oracle(out, what=what)
    n_rows = _assert_packed_kv_halves(case, what)
    assert n_rows == sum(_PACKED_KV_LENS)
    again = _run_with_slots(case, torch.int64)
    assert torch.equal(out, again), f"{what}: two launches differ"
    _mode_pin(case)


def _e4m3_ulp_distance(a, b):
    """|a - b| in e4m3 ulps AT THE LARGER MAGNITUDE, in the e4m3 VALUE domain (fp32 tensors): the spacing ``2^(floor(log2 |x|) - 3)`` of a
    normal value, the subnormal spacing ``2^-9`` below ``2^-6``.  Computed on the HOST in fp32 (a handful of elements): ``torch.exp2`` on a
    CUDA tensor is a jitted (NVRTC) kernel, and NVRTC does not know every part the block serves."""
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    mag = torch.maximum(a.abs(), b.abs()).clamp_min(2.0**-6)
    return (a - b).abs() / torch.exp2(torch.floor(torch.log2(mag)) - 3.0)


def _e4m3_histogram(dist):
    return dict(
        n=int(dist.numel()),
        exact=int((dist == 0).sum()),
        one_ulp=int(((dist > 0) & (dist <= 1.0 + 1e-6)).sum()),
        more=int((dist > 1.0 + 1e-6).sum()),
        max_ulp=float(dist.max()) if dist.numel() else 0.0,
    )


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
def test_fp8_proj_bf16_attention_e4m3_o_is_the_cast_of_the_gated_o_and_tracks_the_gate_quant_model():
    """The mode's e4m3 O, three readings.  (a) THE STORE PIN: the ``o8`` slot is BITWISE ``cvt.rn.satfinite(bf16 gated O * scale_o)`` of the
    block's own gated bf16 O (the ``o`` slot, gated in place) on every row -- one quantization convention, the padded entry's bytes 0.
    (b) THE GATE-AND-QUANTIZE MODEL of a serving stack, ``e4m3(bf16(bf16(O) * sigmoid(G)) * scale_o)`` on the ORACLE's pre-gate O and gate
    (two bf16 roundings; the block's combine route rounds the fp32 gated value ONCE): the histogram of e4m3-ulp distances is printed; the
    one-ulp claim is made on the band |x| >= 4 of the e4m3 value domain (the exponent bands 2^2 and above, where one e4m3 ulp >= 0.5 exceeds
    the chain's ABSOLUTE noise floor: near zero the two chains differ not by a rounding of a shared value but by the kernels' tanh-form
    sigmoid against torch's for strongly negative gates and by the attention's own accumulation order -- inside the band that floor is
    below one ulp, so the two chains differ by at most one code at a rounding boundary); below the band the distances are REPORTED, never
    claimed (the same floor is many ulps of the 2^-9 subnormal spacing; its size in e4m3 units is printed), no tolerance widened; never
    bitwise.  (c) SATURATION through
    ``update_quant_scales``: ``scale_o x 4096`` drives every live byte that reaches the range to +-448 (``satfinite``: no NaN / inf), ``out``
    stays finite; the original spec restored, ``out`` is bitwise the first run (the three device scalars round-trip)."""
    case = _mode_case(("gq", 12, 2), 12, 2, _LENS)
    case.run()
    case.check_against_oracle()
    scale_o = case.quant.scale_o
    _, _, og, _ = case.slab_views()
    o8 = case.o8_view()
    live, dead = case.live, ~case.live
    # (a) the store pin
    assert torch.equal(o8.view(torch.uint8), quant_e4m3(og, scale_o).view(torch.uint8)), "o8 is not the e4m3 cast of the block's own gated O at scale_o"
    assert (o8[dead].view(torch.uint8) == 0).all() and (og[dead] == 0).all(), "a padded entry's gated O / e4m3 O must be exactly 0"
    # (b) the model on the oracle's pre-gate O and gate (the step's rows of every live entry)
    o_ref = torch.stack([case.ref_o[b] for b, L in enumerate(case.kv_lens) if L])
    g_ref = torch.stack([case.ref_gate[b] for b, L in enumerate(case.kv_lens) if L])
    model = quant_e4m3((o_ref.float() * torch.sigmoid(g_ref.float())).to(torch.bfloat16), scale_o)
    a, m = o8[live].float(), model.float()
    dist = _e4m3_ulp_distance(a, m)
    band = (torch.maximum(a.abs(), m.abs()) >= 4.0).cpu()
    full, in_band, below = _e4m3_histogram(dist), _e4m3_histogram(dist[band]), _e4m3_histogram(dist[~band])
    floor = float((a - m).abs().cpu()[~band].max()) if bool((~band).any()) else 0.0  # the chain's absolute noise floor below the band, in e4m3 units
    print(
        f"[mode 24/2 S_q 2 p16] e4m3 O vs the gate-quant model (scale_o {scale_o:.3f}): all {full}; band |x| >= 4: {in_band}; below the band: {below}; "
        f"max |block - model| below the band {floor:.4f} (e4m3 units; one ulp in the band is >= 0.5)"
    )
    assert in_band["n"] > 0.5 * full["n"], "the band must hold most of the live elements for the claim to mean anything"
    assert in_band["more"] == 0, f"an element in the one-ulp band is more than one e4m3 ulp from the gate-quant model: {in_band}"
    assert full["exact"] > 0.5 * full["n"], f"fewer than half the codes agree bitwise with the model: {full}"
    # the oracle's own e4m3 O (the block's quantization point replicated) against the block's: the same histogram shape, reported
    o8_ref = torch.stack([case.ref_o8[b] for b, L in enumerate(case.kv_lens) if L]).float()
    print(f"[mode 24/2 S_q 2 p16] e4m3 O vs the ORACLE's e4m3 O: {_e4m3_histogram(_e4m3_ulp_distance(a, o8_ref))}")
    # (c) saturation through a recalibration, then the round-trip
    q = case.quant
    out0 = case.out.clone()
    case.blk.update_quant_scales(QuantSpec.bf16_attention(q.descale_h, q.descale_w_qkvg, q.descale_w_o, q.scale_o * 4096.0))
    case.execute()
    torch.cuda.synchronize()
    o8s = case.o8_view().float()
    assert torch.isfinite(o8s).all() and (o8s.abs() <= 448.0).all(), "satfinite: never NaN / inf, never past +-448"
    sat = (o8s[live].abs() == 448.0).float().mean().item()
    print(f"[mode 24/2 S_q 2 p16] scale_o x 4096: {sat:.3f} of the live bytes saturate at +-448; out finite {bool(torch.isfinite(case.out.float()).all())}")
    assert sat > 0.5 and torch.isfinite(case.out.float()).all() and (case.out[dead] == 0).all()
    case.blk.update_quant_scales(q)
    case.execute()
    torch.cuda.synchronize()
    assert torch.equal(case.out, out0), "the original scales restored must reproduce the first run bitwise"


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
def test_fp8_proj_bf16_attention_get_workspace_size_is_honest_and_the_carve_has_no_compact_qkv():
    """The ``get_workspace_size()`` twin under the mode: the split partials inside, a sentinel tail untouched by an execute, the carve
    without the compact e4m3 Q / K / V slots and with ``o8``; two launches bitwise."""
    case = _mode_case(("std", 16, 2), 16, 2, _LENS)
    blk = case.blk
    lay = blk._layout()
    assert lay.q8 == lay.k8 == lay.v8 == -1 and lay.o8 >= 0
    scratch = blk._sdpa.scratch_workspace_bytes()
    if _cc() == _SM107:
        assert blk.decode_split_kv >= 2 and scratch > 0, (blk.decode_split_kv, scratch)
    assert blk.get_workspace_size() >= lay.total_bytes + scratch
    need = blk.get_workspace_size()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    out = torch.empty_like(case.out)
    case.execute(out=out, ws=ws)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.check_against_oracle(out, what="mode, sentinel-tailed workspace")
    a = case.run().clone()
    b = case.run().clone()
    assert torch.equal(a, b) and torch.equal(a, out)


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
def test_fp8_proj_bf16_attention_capture_of_the_first_execute_replays_bitwise_and_takes_new_inputs():
    """The capture cell's twin under the mode: the FIRST execute of a fresh declaration captured on a NEW stream with no warm-up (the FP8
    forced-tile GEMMs on the jit-only route: no handle at the first execute; the first-execute scalar write captured WITH its values), the
    replay bitwise the eager run, a new step (lengths / table / slots / e4m3 inputs under the SAME scales) through the captured tensors."""
    _capture_replay_cell(quant=True, **_mode_part_kw())


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
def test_fp8_proj_bf16_attention_update_quant_scales_replays_the_live_values_after_a_warmed_up_capture():
    """``update_quant_scales`` under the mode writes the THREE scalars (``alpha_qkvg`` / ``alpha_o`` / ``scale_o``) in place: a graph captured
    after one eager warm-up replays with the LIVE values (bitwise the eager run after the update; the e4m3 point moved, so the output moved
    too, and it matches the oracle at the new ``scale_o``); the plan facts refused by name."""
    case = _Case(8, 1, (100, 384, 0, 500), seed=31, quant=True, blk_kw=_mode_part_kw()).compile()
    case.execute()  # the warm-up: the first-execute scalar write lands eagerly
    torch.cuda.synchronize()
    out0 = case.out.clone()
    case.check_against_oracle(out0, what="mode, warm-up")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            case.execute()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(case.out, out0), "the replay of a warmed-up capture differs from the eager run"
        q = case.quant
        half = QuantSpec.bf16_attention(q.descale_h, q.descale_w_qkvg, q.descale_w_o, q.scale_o * 0.5)
        case.blk.update_quant_scales(half)
        assert case.blk.quant is half
        graph.replay()
        torch.cuda.synchronize()
        rep = case.out.clone()
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, rep), "the replay did not read the live scalars"
        assert not torch.equal(rep, out0), "a halved scale_o moves the e4m3 point: the output must move"
        ref_half = case.recompute_oracle(q.scale_o * 0.5)
        c = _cos(rep[case.live], ref_half[case.live])
        print(f"[mode 8/1 S_q 1] replay after update_quant_scales(scale_o / 2) vs the oracle at that scale: cos {c:.6f}")
        assert c > 0.999 and (rep[~case.live] == 0).all()
        with pytest.raises(ValueError, match="QuantSpec.attention_dtype is a plan fact"):
            case.blk.update_quant_scales(QuantSpec(q.descale_h, q.descale_w_qkvg, q.descale_w_o, 1.0, 1.0, 1.0, q.scale_o))
        with pytest.raises(ValueError, match="is unused under attention_dtype=torch.bfloat16"):
            case.blk.update_quant_scales(
                QuantSpec(q.descale_h, q.descale_w_qkvg, q.descale_w_o, 1.0, None, None, q.scale_o, torch.float8_e4m3fn, torch.bfloat16)
            )
        case.blk.update_quant_scales(q)
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, out0)
    finally:
        graph.reset()


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
@pytest.mark.parametrize("page", SERVING_PAGE_SIZES)
def test_fp8_proj_bf16_attention_l1_l2_l3_pools_are_bitwise_at_every_page_size(page):
    """The pool-layout matrix under the mode (bf16 pools): the three stride patterns holding the same values give a BITWISE ``out`` at pages
    16 / 32 / 64 / 128, each within the oracle's budget."""
    outs = {}
    for layout in _LAYOUTS:
        case = _mode_case(("layout", page, layout), 12, 2, (200, 0, 77, 1000), page=page, layout=layout, seed=3)
        outs[layout] = case.run().clone()
        case.check_against_oracle(what=f"mode {layout} p{page}")
    for layout in _LAYOUTS[1:]:
        assert torch.equal(
            outs[layout], outs["hnd"]
        ), f"{layout} != hnd at page {page}: max|diff| {(outs[layout].float() - outs['hnd'].float()).abs().max().item():.3e}"


# The MEASURED launch counts of the mode's DECODE chain per served part -- the 24 / 2 geometry at d_model 2560, B 2 x KV 512, S_q 1, the split
# ladder pinned at decode_split_kv=4 (the decode tile + its combine on every part: the chain the docs describe; the split count sizes the grid,
# never the kernel list) -- by (fuse_cache_write_small_m, fuse_gate).  The census cell below is the detector (EVERY kernel the profiler sees is
# counted, a GEMM's split-K reduce included) and the docs quote these numbers; a tree that changes a stage's kernel count changes this table
# and the prose with it.  cc 10.0 serves the gate as its own launch only (fuse_gate=True is the part's typed decline): the two write-through arms.
_MODE_DECODE_LAUNCHES = {
    _SM107: {(False, True): 7, (True, True): 6, (False, False): 8, (True, False): 7},
    _SM100: {(False, False): 8, (True, False): 7},
}


@requires_dense_part
@requires_fp8_proj_bf16_attention_part
def test_fp8_proj_bf16_attention_launch_census_is_the_bf16_chain_plus_the_quantize_launch():
    """The DECODE chain's CUPTI census under the mode on every served part, against the bf16 UNFUSED chain of the same declaration, per
    write-through arm and -- on cc 10.7 -- per gate placement.  EVERY kernel the profiler sees is counted (a GEMM's split-K reduce included;
    the names are printed): the quantize kernel present exactly once and only under the mode, the cache-write kernel exactly on the separate
    arm, the gate's own kernel exactly on the separate-gate arm, no hidden copy, the fold dropping exactly one launch with ``out`` bitwise
    across it, and the TOTAL per arm = ``_MODE_DECODE_LAUNCHES`` (the number the docs publish).  A LAUNCH-COUNT cell, not a numerics cell:
    declared small (B 2 x KV 512 at the 24 / 2 geometry, ``decode_split_kv=4`` pinned so the counted chain -- the decode tile + its combine --
    does not depend on the part's split model) and built with ``check_numerics=False`` (no reference: the oracle matrix cells own the numerics;
    the full-sequence fp8 reference at long KV made the earlier form of this cell too heavy for the smoke tier), so it stays in the module's
    L0: on the development parts it costs ~40 s wall / 840 MiB peak allocated (cc 10.7, 204 SMs) and ~140 s / 520 MiB (cc 10.0, 148 SMs), the
    eight chains' compiles included (measured with the compiled-plan cache disabled), against ~2.5 min and ~1.6 GiB per fp32 logits tensor for
    the oracle form it replaces.  Batch 2, not 1: at the current develop the dense-layout binding's GATE predicate
    refuses the batch-1 singleton-stride slab column on cc 10.7 (a defect outside this change); the chain has no batch-dependent stage."""
    part = _cc()
    arms = _MODE_DECODE_LAUNCHES[part]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cells = {}
    for cw, fg in arms:
        bk = dict(fuse_cache_write_small_m=cw, fuse_gate=fg, decode_split_kv=4)
        cells[("mode", cw, fg)] = _Case(12, 1, (512, 512), seed=1, d_model=2560, quant=True, blk_kw=bk, check_numerics=False).compile()
        cells[("bf16", cw, fg)] = _Case(12, 1, (512, 512), seed=1, d_model=2560, blk_kw=dict(fuse_norm_rope=False, **bk), check_numerics=False).compile()
    census = {k: _cupti_kernels(lambda c=c: c.execute()) for k, c in cells.items()}
    if any(k is None for k, _ in census.values()):
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    for cw, fg in arms:  # every arm's names first (a count that moved is read off one log), the pins after
        k_mode, m_mode = census[("mode", cw, fg)]
        k_bf, _ = census[("bf16", cw, fg)]
        blk = cells[("mode", cw, fg)].blk
        print(
            f"\n[mode census cc {part[0]}.{part[1]}] fuse_cache_write_small_m={cw} fuse_gate={fg} (readbacks: cache_write_in_norm_rope="
            f"{blk.cache_write_in_norm_rope}, gate_fused={blk.gate_fused}, decode_split_kv={blk.decode_split_kv}): {len(k_mode)} kernels + "
            f"{len(m_mode)} memset/memcpy:\n  " + "\n  ".join(n[:120] for n in k_mode)
        )
        print(f"bf16 unfused chain, the same arms: {len(k_bf)} kernels:\n  " + "\n  ".join(n[:120] for n in k_bf))
    for cw, fg in arms:
        k_mode, m_mode = census[("mode", cw, fg)]
        k_bf, _ = census[("bf16", cw, fg)]
        blk = cells[("mode", cw, fg)].blk
        assert (blk.decode_split_kv, blk.gate_fused, blk.cache_write_in_norm_rope) == (4, fg, cw), "the declared arm is the one that compiled"
        assert sum("frost_quantize_fp8" in n for n in k_mode) == 1 and not any("frost_quantize" in n for n in k_bf)
        assert any("frost_cache_write" in n for n in k_mode) == (not cw) and any("frost_qk_norm_rope" in n for n in k_mode)
        assert sum("frost_elementwise_gate" in n for n in k_mode) == (0 if fg else 1), "the gate's own launch exactly on the separate-gate arm"
        assert not [n for n in m_mode if "memcpy" in n.lower()], "a hidden copy on the mode's execute path"
        assert len(k_mode) == arms[(cw, fg)], f"the mode's launch count at this arm is {len(k_mode)}, the published {arms[(cw, fg)]}: {k_mode}"
    for fg in sorted({fg for _, fg in arms}):
        assert len(census[("mode", True, fg)][0]) == len(census[("mode", False, fg)][0]) - 1, "the write-through fold drops exactly the cache-write launch"
        assert torch.equal(_bits(cells[("mode", True, fg)].out), _bits(cells[("mode", False, fg)].out))
    torch.cuda.synchronize()
    print(
        f"[mode census cc {part[0]}.{part[1]}] cost: {time.perf_counter() - t0:.1f} s wall (declarations, compiles, the profiled launches), "
        f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated"
    )
