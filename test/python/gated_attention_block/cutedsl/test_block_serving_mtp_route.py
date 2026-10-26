# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's DECODE route for the speculative (MTP) verify step -- table-driven, one KV stream per sequence
where a kernel offers it; the MTP verify step to ``S_q = 8`` on the one- and two-stream routes, the decode tile UNPACKED past the
packed routes up to ``S_q = 32`` tokens, a typed decline past 32.

The route (``cudnn.gated_attention_block.decode_route``) reads ONE table, ``config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES`` --
the Rubin decode tile's widths that take a step in ONE token unit (the KV streamed once per sequence and KV head) -- and applies
ONE rule under the default policy (``decode_route_policy="single_unit"``): packed rows ``S_q x group`` within the widest width
ride that width in one unit (the table is (16, 32, 64): the 64-column tile takes a packed step of up to 64 rows); past the table the
step rides the paged d256 PREFILL tile with the whole group packed (one Q tile per sequence and KV head: the KV streamed once)
wherever that tile packs the group (a group dividing its 128 rows); a group the prefill tile cannot pack (12 = 24 / 2) keeps the
decode tile's token-unit arm -- up to two units of the 64-column tile, two streams -- and the unpacked tile (one head per CTA) past
it, the fewest streams any kernel offers it today.  The policy ``"token_units"`` is the SDPA adapter's own route verbatim (the widest
tile in up to two token units, then the unpacked tile), the A/B's other arm; ``decode_q_tile_max=32`` caps the table at the measured
widths (the route before the 64-column width landed, value for value: the A/B's "develop behaviour" arm).  Past the packed routes
(the single-unit table, the prefill tile's 128 packed rows -- ``DECODE_PREFILL_TILE_ROWS`` --, two token units) the step rides the
decode tile UNPACKED, one head per CTA (``group`` KV streams: declared, unmeasured, the arm every group had before the table landed),
up to ``DECODE_MAX_ROWS = DECODE_MAX_S_Q = 32`` tokens; past 32 the step is a typed decline (the ``decode_rows`` row).

Host cells (any device): the table and the rule, the cap restoring the former route, the adapter's route under the cap equal to the
shared SM100 rule with the Rubin constants at every ``(S_q, group)``, the route per policy at the three serving geometries of the
MTP cells (24 / 2, 32 / 2, 8 / 1) and the corner groups, the unpacked fallback from ``S_q`` 9 to 32 at every served group (32 / 2
pinned cell by cell) with the decline past 32 and under a cap held below the unpacked width, the route under ``decode_q_tile_max=32``
+ ``token_units`` equal to the former rule at every ``(group, S_q <= 32)``, the declaration facts on the prefill route (the THD
stage, the split, the gate in the paged body's epilogue) and the cap knob, the prefill route's Q lengths as a workspace slot (no
plan-owned tensor), the envelope per policy.  Rubin cells: every MTP cell vs the per-sequence fp32 oracle on its route (the prefill
route at 32 / 2 ``S_q`` 5 / 8, the 64-column tile in one unit at 32 / 2 ``S_q`` 4, 24 / 2 ``S_q`` 4 and 8 / 1 ``S_q`` 5 / 8, in two units
at 24 / 2 ``S_q`` 8; the UNPACKED arm at 32 / 2 ``S_q`` 12 and 32, 16 streams) with a padded entry, the prefill route vs the two-unit
route on IDENTICAL inputs at ``S_q = 5``, two launches
bitwise, the workspace honest, CUDA-graph capture of the FIRST execute at ``S_q = 8`` and its replay over new lengths / tables /
slots / inputs, an EAGER execute after a captured first execute with no replay in between.
"""

import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    DECODE_CLAIMED_MAX_N_Q,
    DECODE_CLAIMED_MAX_S_Q,
    DECODE_MAX_ROWS,
    DECODE_MAX_S_Q,
    DECODE_PREFILL_ROUTE_MIN_CTAS,
    DECODE_PREFILL_TILE_ROWS,
    DECODE_ROUTE_POLICIES,
    DECODE_ROUTE_SINGLE_UNIT,
    DECODE_ROUTE_TOKEN_UNITS,
    DecodeRoute,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    decode_route,
)
from cudnn.gated_attention_block.api import DECODE_TILE, DEFAULT_DECODE_ROUTE_POLICY, PREFILL_TILE, _Sdpa, decode_split_choice

requirement_error = cutedsl_requirement_error("the gated attention block's MTP route tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_block_serving_decode import _GROUPS, _Case, _bits, _cos, _geom, _make_block, _written  # noqa: E402

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SM107 = (10, 7)
_D = 256
_ROPE = 64
_D_MODEL = 512
_MTP_MAX_S_Q = 8  # the MTP verify step the route tables below are spelled to (an MTP-7 step); the served envelope runs to DECODE_MAX_S_Q = 32

# (tile, n_q, token_units, pack_gqa, kv_streams, claimed) per (group, S_q) under the DEFAULT policy ("single_unit"), read WITHOUT a
# batch (= below the prefill route's fill) -- the Rubin table (16, 32, 64), then two units of the 64-column tile wherever two hold the
# step.  The prefill tile packs the groups that divide 128 but is ONE unsplit CTA per (sequence, KV head), so the default policy takes
# it only where the step FILLS the part (``batch x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS`` = 64 CTAs: ``_FILLED`` below, read at
# ``_fill_batch(group)`` sequences) or where no two-unit route holds the step.  Group 12 (24 / 2) cannot pack on the prefill tile, so
# past the table it keeps the token-unit arm at any batch.  "claimed" = the packed decode tile at S_q <= 4 on the MEASURED widths
# (16 / 32): a 64-column cell is routed and declared, never claimed, until its perf row lands.  Past the packed routes (S_q 9 at 32 / 2:
# 144 packed rows, past the prefill tile and a third token unit) the step rides the decode tile UNPACKED -- one head per CTA, ``group``
# KV streams -- up to 32 tokens (the 16-column tile to S_q 16, the 32-column tile past it); 33 is the decline.
_P, _T = PREFILL_TILE, DECODE_TILE
_SINGLE_UNIT = {
    16: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 32, 1, True, 1, True),
        3: (_T, 64, 1, True, 1, False),
        4: (_T, 64, 1, True, 1, False),
        5: (_T, 64, 2, True, 2, False),
        8: (_T, 64, 2, True, 2, False),
        9: (_T, 16, 1, False, 16, False),
        12: (_T, 16, 1, False, 16, False),
        16: (_T, 16, 1, False, 16, False),
        17: (_T, 32, 1, False, 16, False),
        32: (_T, 32, 1, False, 16, False),
        33: (None, 0, 0, False, 0, False),
    },
    12: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 32, 1, True, 1, True),
        3: (_T, 64, 1, True, 1, False),
        4: (_T, 64, 1, True, 1, False),
        5: (_T, 64, 1, True, 1, False),
        6: (_T, 64, 2, True, 2, False),
        8: (_T, 64, 2, True, 2, False),
        9: (_T, 64, 2, True, 2, False),
        10: (_T, 64, 2, True, 2, False),
        11: (_T, 16, 1, False, 12, False),
        32: (_T, 32, 1, False, 12, False),
        33: (None, 0, 0, False, 0, False),
    },
    8: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 16, 1, True, 1, True),
        3: (_T, 32, 1, True, 1, True),
        4: (_T, 32, 1, True, 1, True),
        5: (_T, 64, 1, True, 1, False),
        8: (_T, 64, 1, True, 1, False),
        9: (_T, 64, 2, True, 2, False),
        16: (_T, 64, 2, True, 2, False),
        17: (_T, 32, 1, False, 8, False),
        32: (_T, 32, 1, False, 8, False),
        33: (None, 0, 0, False, 0, False),
    },
    6: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 16, 1, True, 1, True),
        4: (_T, 32, 1, True, 1, True),
        5: (_T, 32, 1, True, 1, False),
        6: (_T, 64, 1, True, 1, False),
        8: (_T, 64, 1, True, 1, False),
        10: (_T, 64, 1, True, 1, False),
        11: (_T, 64, 2, True, 2, False),
        20: (_T, 64, 2, True, 2, False),
        21: (_T, 32, 1, False, 6, False),
        32: (_T, 32, 1, False, 6, False),
        33: (None, 0, 0, False, 0, False),
    },
    4: {
        1: (_T, 16, 1, True, 1, True),
        4: (_T, 16, 1, True, 1, True),
        5: (_T, 32, 1, True, 1, False),
        8: (_T, 32, 1, True, 1, False),
        9: (_T, 64, 1, True, 1, False),
        16: (_T, 64, 1, True, 1, False),
        17: (_T, 64, 2, True, 2, False),
        32: (_T, 64, 2, True, 2, False),
        33: (None, 0, 0, False, 0, False),
    },
}
# The default policy AT the fill: the prefill tile on the cells it packs past the single-unit table (every other cell is _SINGLE_UNIT's).
_PREFILL = (_P, 0, 0, True, 1, False)
_FILLED = {16: {5: _PREFILL, 8: _PREFILL}, 8: {9: _PREFILL, 16: _PREFILL}, 4: {17: _PREFILL, 32: _PREFILL}}


def _fill_batch(group):
    """The smallest batch at which a step of ``group`` fills the part for the prefill route (``batch x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS``)."""
    return -(-DECODE_PREFILL_ROUTE_MIN_CTAS // _GROUPS[group][1])


# The same cells under "token_units" (the adapter's route verbatim, inside the served envelope): two units of the 64-column tile wherever
# two hold the step, whatever the batch -- the default policy's reading below the fill; the unpacked arm is the same.
_TOKEN_UNITS = {
    16: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 32, 1, True, 1, True),
        3: (_T, 64, 1, True, 1, False),
        4: (_T, 64, 1, True, 1, False),
        5: (_T, 64, 2, True, 2, False),
        8: (_T, 64, 2, True, 2, False),
        9: (_T, 16, 1, False, 16, False),
        16: (_T, 16, 1, False, 16, False),
        17: (_T, 32, 1, False, 16, False),
        32: (_T, 32, 1, False, 16, False),
        33: (None, 0, 0, False, 0, False),
    },
    12: _SINGLE_UNIT[12],
    8: {**_SINGLE_UNIT[8], 9: (_T, 64, 2, True, 2, False), 16: (_T, 64, 2, True, 2, False)},
    6: _SINGLE_UNIT[6],
    4: {**_SINGLE_UNIT[4], 17: (_T, 64, 2, True, 2, False), 32: (_T, 64, 2, True, 2, False)},
}
_TABLES = {DECODE_ROUTE_SINGLE_UNIT: _SINGLE_UNIT, DECODE_ROUTE_TOKEN_UNITS: _TOKEN_UNITS}


def _check_route(r: DecodeRoute, want, *, group, s_q, policy):
    tile, n_q, units, packed, streams, claimed = want
    assert isinstance(r, DecodeRoute) and (r.s_q, r.group, r.policy) == (s_q, group, policy), r
    assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams, r.claimed) == want, (group, s_q, policy, r)
    assert r.served is (tile is not None) and r.decode_tile is (tile == _T) and r.prefill_tile is (tile == _P), r
    assert r.pdl_capable is (tile == _T), ("stage (4) takes the programmatic-dependent-launch switch on the decode tile only", r)
    assert (r.reason == "") is r.served and (r.served or "EXTEND form" in r.reason), r
    if tile == _T and packed:
        assert units == -(-s_q // (n_q // group)) and streams == units, r
    elif tile == _T:
        assert units == 1 and s_q <= n_q and streams == (1 if group == 1 else group), r
    elif tile == _P:
        assert units == 0 and streams == 1 and packed and s_q * group <= DECODE_PREFILL_TILE_ROWS and 128 % group == 0, r


# ---------------------------------------------------------------------------
# The table, the rule, the adapter's route
# ---------------------------------------------------------------------------


def test_the_single_unit_table_and_the_rule():
    """ONE ordered table of single-unit widths (16, 32, 64); THE rule = the narrowest width of the FORM's table holding the packed
    rows, else 0 (past the table) -- the 64-column width for a packed group of >= 2 heads over a dense cache or over pages under the
    causal / window mask (``decode_d256_wide_tile_admits``), the (16, 32) table for an unpacked step and a mask-free paged step; the
    routed maximum is the table's widest entry and never exceeds the width the Rubin tile compiles for."""
    from cudnn.sdpa.fwd import config_sm107

    table = config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES
    assert table == (16, 32, 64) and list(table) == sorted(table) and all(w > 0 for w in table)
    assert config_sm107.D256_DECODE_ROUTED_MAX_Q_ROWS == table[-1] <= config_sm107.D256_DECODE_MAX_Q_ROWS_SM107 == 64
    assert config_sm107.D256_DECODE_ROUTED_MAX_TOKEN_UNITS == 2  # the adapter's token-unit cap, unchanged
    rule = config_sm107.decode_d256_single_unit_q_tile
    for s_q in range(0, 70):
        for g in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 33, 64):
            rows = s_q * g
            for paged, masked in ((True, True), (False, False), (True, False)):
                widths = config_sm107.decode_d256_route_widths(g, paged=paged, masked=masked)
                assert widths == (table if config_sm107.decode_d256_wide_tile_admits(g, paged, masked) else table[:2]), (g, paged, masked)
                want = 0 if (s_q <= 0 or g <= 0) else next((w for w in widths if rows <= w), 0)
                assert rule(s_q, g, paged=paged, masked=masked) == want, (s_q, g, paged, masked, rule(s_q, g, paged=paged, masked=masked), want)
            assert rule(s_q, g) == rule(s_q, g, paged=False, masked=True)  # the defaults: a dense cache (any mask)
    assert rule(1, 12) == 16 and rule(2, 12) == 32 and rule(3, 12) == 64 and rule(5, 12) == 64 and rule(6, 12) == 0 and rule(1, 32) == 32 and rule(2, 32) == 64
    assert rule(32, 1) == 32 and rule(33, 1) == 0 and rule(16, 1) == 16  # unpacked: the 64-column width's predicate declines it
    assert rule(4, 16, paged=True, masked=False) == 0 and rule(4, 16, paged=True, masked=True) == 64 and rule(4, 16, paged=False, masked=False) == 64
    assert rule(4, 12, max_q_tile=32) == 0 and rule(2, 12, max_q_tile=32) == 32 and rule(2, 12, max_q_tile=16) == 0
    assert config_sm107.decode_d256_wide_tile_admits(2, True, True) and not config_sm107.decode_d256_wide_tile_admits(1, False, True)
    assert not config_sm107.decode_d256_wide_tile_admits(16, True, False) and config_sm107.decode_d256_wide_tile_admits(16, False, False)


def test_the_64_column_width_is_in_the_table_and_the_cap_restores_the_former_route():
    """The 64-column entry of the table routes the 24 / 2 step at ``S_q = 4`` -- 48 rows -- in ONE unit through the unchanged rule and
    the adapter's route; ``S_q`` 6 .. 8 (72 .. 96 rows) in TWO units of it (the adapter's token-unit arm on the widest width);
    ``max_q_tile=32`` / ``decode_q_tile_max=32`` restores the route before the width landed, value for value, and reads back."""
    from cudnn.sdpa.fwd import config_sm107

    assert config_sm107.decode_d256_single_unit_q_tile(4, 12) == 64 and config_sm107.decode_d256_single_unit_q_tile(5, 12) == 64
    assert config_sm107.decode_d256_single_unit_q_tile(6, 12) == 0 and config_sm107.decode_d256_single_unit_q_tile(2, 12) == 32
    assert config_sm107.decode_d256_q_tile(4, 12) == 64 and config_sm107.decode_d256_q_tile(4, 16) == 64  # one unit of the width
    assert config_sm107.decode_d256_q_tile(8, 12) == 64  # 96 rows: two units of the WIDEST width (the adapter's token-unit arm)
    assert config_sm107.decode_d256_q_tile(11, 12) == 0  # three units: past the adapter's cap
    assert config_sm107.decode_d256_q_tile(4, 12, max_q_tile=32) == 32 and config_sm107.decode_d256_q_tile(5, 12, max_q_tile=32) == 0
    geom, _ = _geom(12)
    r = decode_route(geom, 4)
    assert (r.tile, r.n_q, r.token_units, r.kv_streams, r.claimed, r.q_tile_max) == (DECODE_TILE, 64, 1, 1, False, None), r
    r8 = decode_route(geom, 8)
    assert (r8.tile, r8.n_q, r8.token_units, r8.kv_streams, r8.pack_gqa) == (DECODE_TILE, 64, 2, 2, True), r8
    c = decode_route(geom, 4, max_q_tile=32)
    assert (c.tile, c.n_q, c.token_units, c.kv_streams, c.claimed, c.q_tile_max) == (DECODE_TILE, 32, 2, 2, True, 32), c
    c8 = decode_route(geom, 8, max_q_tile=32)
    assert (c8.tile, c8.n_q, c8.token_units, c8.kv_streams, c8.pack_gqa) == (DECODE_TILE, 16, 1, 12, False), c8  # the unpacked arm of the former route
    for bad in (48, 0, 128, True, "64"):
        with pytest.raises(ValueError, match="decode_q_tile_max must be one of"):
            decode_route(geom, 4, max_q_tile=bad)


def test_the_adapters_route_under_the_cap_is_the_former_rule_verbatim():
    """``config_sm107.decode_d256_q_tile(max_q_tile=32)`` equals the shared SM100 rule with the Rubin constants (16, 32) at every
    ``(S_q, group)``: the cap is the route before the 64-column width landed, value for value -- the A/B's "develop behaviour" arm.
    ``max_token_units=1`` is the single-unit rule alone, whatever the table."""
    from cudnn.sdpa.fwd import config_sm100, config_sm107

    for s_q in range(0, 70):
        for g in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 33, 64):
            old = config_sm100.decode_d256_q_tile(s_q, g, 32, 2 if g > 1 else 1)
            assert config_sm107.decode_d256_q_tile(s_q, g, max_q_tile=32) == old, (s_q, g, config_sm107.decode_d256_q_tile(s_q, g, max_q_tile=32), old)
            assert config_sm107.decode_d256_q_tile(s_q, g, max_token_units=1) == config_sm107.decode_d256_single_unit_q_tile(s_q, g), (s_q, g)
            assert config_sm107.decode_d256_q_tile(s_q, g, max_token_units=1, max_q_tile=32) == config_sm107.decode_d256_single_unit_q_tile(
                s_q, g, max_q_tile=32
            )
    assert config_sm107.decode_d256_q_tile(4, 12, max_q_tile=32) == 32 and config_sm107.decode_d256_q_tile(5, 12, max_q_tile=32) == 0
    assert config_sm107.decode_d256_q_tile(4, 12) == 64 and config_sm107.decode_d256_q_tile(5, 12) == 64 and config_sm107.decode_d256_q_tile(4, 16) == 64


# ---------------------------------------------------------------------------
# The route per policy, the envelope, the reject cells
# ---------------------------------------------------------------------------


def test_constants_and_policies():
    # the served bound is the former one, in the former meaning: 32 tokens per sequence (the 32-column tile unpacked); the prefill
    # route's 128 packed rows carry their own name; the claimed bound stays 4 on the measured widths
    assert (DECODE_CLAIMED_MAX_S_Q, DECODE_MAX_S_Q, DECODE_MAX_ROWS, DECODE_CLAIMED_MAX_N_Q, DECODE_PREFILL_TILE_ROWS) == (4, 32, 32, 32, 128)
    assert DECODE_MAX_S_Q == DECODE_MAX_ROWS and _MTP_MAX_S_Q < DECODE_MAX_S_Q
    assert DECODE_ROUTE_POLICIES == (DECODE_ROUTE_SINGLE_UNIT, DECODE_ROUTE_TOKEN_UNITS) == ("single_unit", "token_units")
    assert DEFAULT_DECODE_ROUTE_POLICY == DECODE_ROUTE_SINGLE_UNIT
    assert (DECODE_TILE, PREFILL_TILE) == ("decode_d256_f16", "prefill_d256_f16")
    from cudnn.sdpa.fwd.config_sm100 import CfgD256
    from cudnn.sdpa.fwd.config_sm107 import D256_DECODE_SINGLE_UNIT_Q_TILES

    assert DECODE_PREFILL_TILE_ROWS == CfgD256.TILE_M  # the prefill tile's Q rows: one packed Q tile per (sequence, KV head)
    assert DECODE_MAX_ROWS == max(w for w in D256_DECODE_SINGLE_UNIT_Q_TILES if w <= 32)  # the widest width the UNPACKED form rides


@pytest.mark.parametrize("policy", DECODE_ROUTE_POLICIES)
@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_the_route_per_policy_at_every_served_group(group, policy):
    """Which tile each ``S_q`` rides, per policy: the single-unit table, the packed prefill tile where the group divides its rows, the
    token-unit / unpacked arms where it does not (and everywhere under ``token_units``); ``kv_streams`` names the KV streams per
    (sequence, KV head); ``claimed`` is the packed DECODE tile at ``S_q <= 4`` only; the default policy is ``single_unit``."""
    geom, _ = _geom(group)
    fill = _fill_batch(group)
    for s_q, want in _TABLES[policy][group].items():
        _check_route(decode_route(geom, s_q, policy=policy), want, group=group, s_q=s_q, policy=policy)
        assert decode_route(geom, s_q, policy=policy).batch is None
        if policy == DEFAULT_DECODE_ROUTE_POLICY:
            assert decode_route(geom, s_q) == decode_route(geom, s_q, policy=policy)
            f = decode_route(geom, s_q, policy=policy, batch=fill)
            _check_route(f, _FILLED.get(group, {}).get(s_q, want), group=group, s_q=s_q, policy=policy)
            assert f.batch == fill and decode_route(geom, s_q, policy=policy, batch=fill - 1) == decode_route(geom, s_q, policy=policy), (group, s_q, f)
        else:
            assert decode_route(geom, s_q, policy=policy, batch=fill) == decode_route(geom, s_q, policy=policy), "token_units never takes the prefill tile"
    for s_q in range(1, DECODE_MAX_S_Q + 1):
        r = decode_route(geom, s_q, policy=policy)
        assert r.served and r.kv_streams >= 1 and (r.decode_tile or r.prefill_tile), r
        if policy == DECODE_ROUTE_SINGLE_UNIT and 128 % group == 0 and s_q <= _MTP_MAX_S_Q:
            at_fill = decode_route(geom, s_q, policy=policy, batch=fill)
            assert at_fill.kv_streams == 1, ("a group the prefill tile packs streams the KV once at every MTP step once the step fills the part", at_fill)
            assert r.kv_streams <= 2, ("below the fill at most the two-unit arm's streams", r)
        if policy == DECODE_ROUTE_SINGLE_UNIT:
            assert r.kv_streams <= decode_route(geom, s_q, policy=DECODE_ROUTE_TOKEN_UNITS).kv_streams, ("never more streams than the adapter's route", r)
        if not r.pack_gqa:
            # the UNPACKED arm: one head per CTA, the step's own tokens on the narrowest width holding them, the group's streams
            assert r.decode_tile and r.token_units == 1 and r.kv_streams == group and r.n_q == (16 if s_q <= 16 else 32) and not r.claimed, r


def test_the_corner_groups():
    """MHA stays the unpacked 16-column tile (one stream: one head per KV head); group 32 (64 / 2) packs on the 32-column tile at ``S_q = 1``,
    on the 64-column tile at ``S_q = 2`` and on the prefill tile up to 128 rows (two units of the 64-column tile under ``token_units``);
    group 64 rides the 64-column tile at ``S_q = 1`` and the prefill tile at ``S_q = 2``; an odd group (3) packs on the decode tile at
    every ``S_q <= 8`` (whole-group packing has no divisibility rule there)."""
    mk = lambda h_q, h_kv: GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=h_q, h_kv=h_kv, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)  # noqa: E731
    mha = mk(4, 4)
    for s_q in range(1, DECODE_MAX_S_Q + 1):
        _check_route(decode_route(mha, s_q), (_T, 16 if s_q <= 16 else 32, 1, False, 1, False), group=1, s_q=s_q, policy=DECODE_ROUTE_SINGLE_UNIT)
    assert not decode_route(mha, DECODE_MAX_S_Q + 1).served
    g32 = mk(64, 2)
    _check_route(decode_route(g32, 1), (_T, 32, 1, True, 1, True), group=32, s_q=1, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g32, 2), (_T, 64, 1, True, 1, False), group=32, s_q=2, policy=DECODE_ROUTE_SINGLE_UNIT)
    for s_q in (3, 4):
        # 96 / 128 packed rows: two units of the 64-column tile below the fill, the prefill tile at 32 sequences x 2 KV heads = 64 CTAs
        _check_route(decode_route(g32, s_q), (_T, 64, 2, True, 2, False), group=32, s_q=s_q, policy=DECODE_ROUTE_SINGLE_UNIT)
        _check_route(decode_route(g32, s_q, batch=32), (_P, 0, 0, True, 1, False), group=32, s_q=s_q, policy=DECODE_ROUTE_SINGLE_UNIT)
        _check_route(decode_route(g32, s_q, policy=DECODE_ROUTE_TOKEN_UNITS), (_T, 64, 2, True, 2, False), group=32, s_q=s_q, policy=DECODE_ROUTE_TOKEN_UNITS)
    _check_route(decode_route(g32, 2, max_q_tile=32), (_T, 32, 2, True, 2, True), group=32, s_q=2, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g32, 2, max_q_tile=32, batch=32), (_P, 0, 0, True, 1, False), group=32, s_q=2, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(
        decode_route(g32, 2, policy=DECODE_ROUTE_TOKEN_UNITS, max_q_tile=32), (_T, 32, 2, True, 2, True), group=32, s_q=2, policy=DECODE_ROUTE_TOKEN_UNITS
    )
    _check_route(
        decode_route(g32, 3, policy=DECODE_ROUTE_TOKEN_UNITS, max_q_tile=32), (_T, 16, 1, False, 32, False), group=32, s_q=3, policy=DECODE_ROUTE_TOKEN_UNITS
    )
    # 160 packed rows: past the prefill tile's 128 and a third token unit, so the step rides the tile UNPACKED (32 streams), as before
    # the table landed; the same at 128 / 2 from S_q 3 (64 streams)
    _check_route(decode_route(g32, 5), (_T, 16, 1, False, 32, False), group=32, s_q=5, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g32, 32), (_T, 32, 1, False, 32, False), group=32, s_q=32, policy=DECODE_ROUTE_SINGLE_UNIT)
    assert not decode_route(g32, 33).served
    g64 = mk(128, 2)
    _check_route(decode_route(g64, 1), (_T, 64, 1, True, 1, False), group=64, s_q=1, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g64, 2), (_T, 64, 2, True, 2, False), group=64, s_q=2, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g64, 2, batch=32), (_P, 0, 0, True, 1, False), group=64, s_q=2, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g64, 2, policy=DECODE_ROUTE_TOKEN_UNITS), (_T, 64, 2, True, 2, False), group=64, s_q=2, policy=DECODE_ROUTE_TOKEN_UNITS)
    # under the cap the 32-column tile holds no 64-head unit: no two-unit route, so the prefill tile at any batch
    _check_route(decode_route(g64, 1, max_q_tile=32), (_P, 0, 0, True, 1, False), group=64, s_q=1, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g64, 1, max_q_tile=32, batch=1), (_P, 0, 0, True, 1, False), group=64, s_q=1, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g64, 3), (_T, 16, 1, False, 64, False), group=64, s_q=3, policy=DECODE_ROUTE_SINGLE_UNIT)
    g3 = mk(12, 4)
    for s_q in range(1, _MTP_MAX_S_Q + 1):
        r = decode_route(g3, s_q)
        assert r.decode_tile and r.pack_gqa and r.token_units == 1 and r.n_q == (16 if 3 * s_q <= 16 else 32), r
    d128 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=128, rope_dim=_ROPE, causal_bottom_right=True)
    assert not decode_route(d128, 1).served and "d_head=128" in decode_route(d128, 1).reason
    with pytest.raises(ValueError, match="decode_route_policy must be one of"):
        decode_route(mha, 1, policy="fastest")


def _former_route(geom, s_q, policy):
    """The route BEFORE the table-driven rule (the (16, 32) table, i.e. ``decode_q_tile_max=32``): the packed decode tile in up to two
    token units where the adapter's rule packs the group, else the tile UNPACKED up to 32 tokens, else not served -- re-derived from
    the config's route functions, the way the former ``decode_route`` read them (``token_units`` is that rule verbatim; ``single_unit``
    differs only where the prefill tile packs the group past the table)."""
    from cudnn.sdpa.fwd.config_sm100 import decode_d256_q_units
    from cudnn.sdpa.fwd.config_sm107 import decode_d256_q_tile

    g = geom.h_q // geom.h_kv
    n_q = decode_d256_q_tile(s_q, g, paged=True, masked=True, max_q_tile=32) if g > 1 else 0
    if n_q:
        units = decode_d256_q_units(s_q, g, n_q)
        return (_T, n_q, units, True, units)
    n_q = decode_d256_q_tile(s_q, 1, paged=True, masked=True, max_q_tile=32)
    return (_T, n_q, 1, False, g) if n_q else (None, 0, 0, False, 0)


def test_the_unpacked_fallback_to_32_tokens_and_the_decline_past_it(monkeypatch):
    """The served envelope is the former one: past the packed routes the step rides the decode tile UNPACKED (one head per CTA,
    ``group`` streams) up to ``S_q = DECODE_MAX_ROWS = 32`` tokens -- 32 / 2 at ``S_q`` 9 .. 32 cell by cell (16 columns to 16 tokens,
    32 past), every served group from the first unpacked step --, under both policies; at ``S_q = 33`` the route says not served with
    the reason (the ``decode_rows`` row: the EXTEND form), the declaration raises that row's message and the envelope predicts it
    (``max_s_q == max_rows == 32``, the routes 1 .. 33 with the last one declined, no ``decode_s_q`` row).  Under
    ``decode_q_tile_max=32`` + ``token_units`` the route is the former rule at every ``(group, S_q <= 32)``, value for value, and
    ``decode_q_tile_max=16`` held below the width an unpacked step of 17 .. 32 tokens needs is a typed decline naming the cap (the
    cap is honoured, never widened)."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom16, _ = _geom(16)
    for policy in DECODE_ROUTE_POLICIES:
        for s_q in range(9, DECODE_MAX_S_Q + 1):
            _check_route(decode_route(geom16, s_q, policy=policy), (_T, 16 if s_q <= 16 else 32, 1, False, 16, False), group=16, s_q=s_q, policy=policy)
    for s_q in (9, 16, 17, 32):
        blk, _ = _make_block(16, 1, s_q)
        r = blk.decode_route
        assert r.served and r.decode_tile and not r.pack_gqa and r.kv_streams == 16 and r.n_q == (16 if s_q <= 16 else 32), r
        assert blk._sdpa.decode_route == r and blk._sdpa.paged_dense and not blk._sdpa.thd and blk.decode_split_kv >= 1
    for group in sorted(_GROUPS):
        geom, _ = _geom(group)
        # the first unpacked step under the default policy (None at 4 / 1: the prefill tile packs its group up to 128 rows = 32 tokens)
        first_unpacked = next((s for s in range(1, DECODE_MAX_S_Q + 1) if not decode_route(geom, s).pack_gqa), None)
        for policy in DECODE_ROUTE_POLICIES:
            for s_q in range(1, DECODE_MAX_S_Q + 1):
                r = decode_route(geom, s_q, policy=policy)
                assert r.served, (group, s_q, policy, r)
                if first_unpacked is not None and s_q >= first_unpacked and policy == DECODE_ROUTE_SINGLE_UNIT:
                    assert not r.pack_gqa and r.kv_streams == group and r.n_q == (16 if s_q <= 16 else 32), (group, s_q, r)
                # the former rule under the cap, value for value (token_units IS that rule; single_unit differs on the prefill tile only)
                c = decode_route(geom, s_q, policy=DECODE_ROUTE_TOKEN_UNITS, max_q_tile=32)
                assert (c.tile, c.n_q, c.token_units, c.pack_gqa, c.kv_streams) == _former_route(geom, s_q, DECODE_ROUTE_TOKEN_UNITS), (group, s_q, c)
            r = decode_route(geom, DECODE_MAX_S_Q + 1, policy=policy)
            assert (
                not r.served and r.tile is None and r.n_q == 0 and r.kv_streams == 0 and f"S_q <= {DECODE_MAX_S_Q}" in r.reason and "EXTEND form" in r.reason
            ), r
            assert _former_route(geom, DECODE_MAX_S_Q + 1, policy) == (None, 0, 0, False, 0), "the former rule declined 33 tokens too"
            with pytest.raises(NotImplementedError, match=f"serves S_q <= {DECODE_MAX_S_Q}.*EXTEND form") as e:
                _make_block(group, 2, DECODE_MAX_S_Q + 1, decode_route_policy=policy)
            env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy=policy)
            assert env.decode.served and env.decode.policy == policy and (env.decode.max_s_q, env.decode.max_rows, env.decode.claimed_max_s_q) == (32, 32, 4)
            assert [x.s_q for x in env.decode.routes] == list(range(1, DECODE_MAX_S_Q + 2)) and not env.decode.routes[-1].served
            assert all(x.served for x in env.decode.routes[:-1]) and all(x.policy == policy for x in env.decode.routes)
            assert str(e.value) == env.decode.declines["decode_rows"] and "decode_s_q" not in env.decode.declines
            assert f"S_q <= {DECODE_MAX_S_Q}" in env.decode.declines["decode_rows"]
            # the cap held below the unpacked step's width: a typed decline naming the cap, the envelope predicting it -- wherever no
            # packed route remains (every group under token_units; under single_unit the prefill tile still packs 4 / 1 at 17 tokens)
            capped = decode_route(geom, 17, policy=policy, max_q_tile=16)
            assert decode_route(geom, 16, policy=policy, max_q_tile=16).served
            env16 = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy=policy, decode_q_tile_max=16)
            if policy == DECODE_ROUTE_SINGLE_UNIT and 128 % group == 0 and 17 * group <= DECODE_PREFILL_TILE_ROWS:
                assert capped.prefill_tile and group == 4 and env16.decode.route(17).prefill_tile, capped
            else:
                assert not capped.served and "decode_q_tile_max=16" in capped.reason and "32 columns" in capped.reason, capped
                with pytest.raises(NotImplementedError, match="decode_q_tile_max=16"):
                    _make_block(group, 1, 17, decode_route_policy=policy, decode_q_tile_max=16)
                assert not env16.decode.route(17).served and env16.decode.route(16).served
        blk32, _ = _make_block(group, 2, DECODE_MAX_S_Q)
        assert blk32.decode_route.served and blk32.decode_route.s_q == DECODE_MAX_S_Q
    # 160 packed rows (group 32 at S_q 5): past the prefill tile's 128 and a third unit, so the UNPACKED arm (32 streams); S_q 4 = 128
    # rows served on the prefill tile
    kw = dict(d_model=_D_MODEL, h_q=64, h_kv=2, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    g32 = GatedAttentionBlockGeometry(**kw)
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=_DEV)  # noqa: E731
    kc = torch.empty(4, 2, 16, _D, dtype=torch.bfloat16, device=_DEV)

    def declare(s_q):
        return GatedAttentionBlockFwd(
            mk(1, s_q, _D_MODEL),
            mk(g32.n_qkvg, _D_MODEL),
            mk(_D),
            mk(_D),
            mk(1, s_q, _ROPE),
            mk(1, s_q, _ROPE),
            mk(_D_MODEL, 64 * _D),
            mk(1, s_q, _D_MODEL),
            g32,
            paged_kv_page_size=16,
            kv_source="paged",
            sample_k_cache=kc,
            sample_v_cache=kc.clone(),
            sample_block_table=torch.zeros(1, 4, dtype=torch.int32, device=_DEV),
        )

    blk5 = declare(5)
    assert blk5.decode_route.decode_tile and not blk5.decode_route.pack_gqa and blk5.decode_route.kv_streams == 32 and blk5.decode_route.n_q == 16
    blk = declare(4)  # ONE sequence: below the prefill route's fill, the 128 packed rows ride two units of the 64-column tile
    assert blk.decode_route.decode_tile and blk.decode_route.token_units == 2 and blk.decode_route.kv_streams == 2 and blk.decode_route.batch == 1
    assert decode_route(g32, 4, batch=32).prefill_tile and decode_route(g32, 4, batch=32).s_q * decode_route(g32, 4, batch=32).group == DECODE_PREFILL_TILE_ROWS
    env = GatedAttentionBlockFwd.serving_envelope(g32, torch.bfloat16, _DEV, device_cc=_SM107)
    assert env.decode.served and env.decode.route(5).served and not env.decode.route(5).pack_gqa and env.decode.route(4, batch=32).prefill_tile
    assert env.decode.route(4).decode_tile and env.decode.route(4).token_units == 2
    with pytest.raises(NotImplementedError, match=f"serves S_q <= {DECODE_MAX_S_Q}.*EXTEND form"):
        declare(33)


def test_declaration_facts_on_the_prefill_route_and_the_policy_knob():
    """A DECODE block whose route is the prefill tile builds stage (4) as the EXTEND form's binding at the dense step's envelope: a THD
    stage at ``(num_sequences=B, max_seq_len=S_q)`` carrying the route, unsplit, the group packed, the gate in the paged body's epilogue
    (``gate_fused`` with no separate launch); a pinned split there is a typed decline; ``decode_route_policy`` selects the two-unit tile
    instead, is refused off the DECODE form and must be a known policy; ``decode_q_tile_max`` caps the width table (32 = the route before
    the 64-column width landed) and is refused off the DECODE form too; the stage list is the serving pipeline."""
    # 32 / 2 at S_q 5: 80 rows, past the table -> the prefill tile where the step fills the part (32 sequences x 2 KV heads = 64 CTAs)
    blk, _ = _make_block(16, 32, 5, fuse_gate=True, fuse_norm_rope=True)
    r = blk.decode_route
    assert r.prefill_tile and r.tile == PREFILL_TILE and r.pack_gqa and r.kv_streams == 1 and r.policy == DECODE_ROUTE_SINGLE_UNIT and not r.claimed
    assert r.batch == 32 and r == decode_route(blk.geom, 5, batch=32)
    assert blk.decode_route_policy == DECODE_ROUTE_SINGLE_UNIT and blk.decode_q_tile_max is None and blk.decode_split_kv == 1 and blk.seq_lens_present is True
    st = blk._sdpa
    assert isinstance(st, _Sdpa) and st.thd and not st.paged_dense and st.decode_route == r and st.decode_split_kv == 1 and not st.seq_lens_present
    assert (st.num_sequences, st.max_seq_len, st.cu_seqlens, st.batch, st.seq_len) == (32, 5, False, 32, 5)
    assert blk.gate_fused and blk._gate is None and st.fuse_gate, "the gate rides the paged prefill body's epilogue"
    # 32 x 5 = 160 rows: past the small-M lowering's band (128 rows), so the fused declaration keeps its fork whatever the knob says
    # (the knob governs decode row counts only); the unfused chain on the same route is the fuse_norm_rope=False declaration
    assert [s.name for s in blk._stages] == ["qkv_gate_proj_norm_rope", "sdpa", "out_proj"] and blk.fuse_norm_rope_effective and not blk.small_m_decode
    blk_f, _ = _make_block(16, 32, 5, fuse_gate=True, fuse_norm_rope=True, fuse_norm_rope_small_m=False)
    assert [s.name for s in blk_f._stages] == ["qkv_gate_proj_norm_rope", "sdpa", "out_proj"] and blk_f.fuse_norm_rope_effective
    blk_l, _ = _make_block(16, 32, 5, fuse_gate=True, fuse_norm_rope=False)
    assert [s.name for s in blk_l._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "out_proj"] and not blk_l.fuse_norm_rope_effective
    assert blk_l.decode_route.prefill_tile and blk_l._sdpa.thd
    assert decode_split_choice(r, batch=32, h_kv=2, max_kv_len=1024, sm_count=204, gated=True, h_q=32) == 1
    # a pinned split on the prefill route: typed; 1 is the unsplit prefill body
    with pytest.raises(NotImplementedError, match="prefill route.*unsplit"):
        _make_block(16, 32, 5, decode_split_kv=2)
    blk1, _ = _make_block(16, 32, 5, fuse_gate=True, decode_split_kv=1)
    assert blk1.decode_route.prefill_tile and blk1.decode_split_kv == 1 and blk1.gate_fused
    # below the fill (4 sequences: 8 CTAs) the SAME declaration rides two units of the 64-column tile, the gate in the split combine,
    # and a pinned split is legal there
    blk_b, _ = _make_block(16, 4, 5, fuse_gate=True, fuse_norm_rope=True)
    rb = blk_b.decode_route
    assert rb.decode_tile and (rb.n_q, rb.token_units, rb.kv_streams, rb.policy, rb.batch, rb.claimed) == (64, 2, 2, DECODE_ROUTE_SINGLE_UNIT, 4, False), rb
    assert blk_b._sdpa.paged_dense and not blk_b._sdpa.thd and blk_b.gate_fused and blk_b.decode_split_kv >= 2 and blk_b._gate is None
    # C0: at decode row counts (4 x 5 = 20 rows) the fused declaration lowers stage (1) to the unfused chain, the paged write-through folded
    # into the norm + RoPE launch (fuse_cache_write_small_m's default below 128 rows; the 160-row chain above keeps the separate launch)
    assert [s.name for s in blk_b._stages] == ["qkv_gate_proj", "qk_norm_rope", "sdpa", "out_proj"] and not blk_b.fuse_norm_rope_effective
    assert blk_b.cache_write_in_norm_rope and blk_b._cache_write is None and not blk_l.cache_write_in_norm_rope
    assert _make_block(16, 4, 5, decode_split_kv=2)[0].decode_split_kv == 2
    # the other arm of the A/B: the two-unit decode tile (two units of the 64-column width) at ANY batch under token_units
    blk_t, _ = _make_block(16, 32, 5, fuse_gate=True, fuse_norm_rope=True, decode_route_policy=DECODE_ROUTE_TOKEN_UNITS, decode_split_kv=2)
    rt = blk_t.decode_route
    assert rt.decode_tile and (rt.n_q, rt.token_units, rt.kv_streams, rt.policy, rt.claimed, rt.batch) == (64, 2, 2, DECODE_ROUTE_TOKEN_UNITS, False, 32)
    assert blk_t._sdpa.paged_dense and not blk_t._sdpa.thd and blk_t._sdpa.decode_route == rt and blk_t.gate_fused and blk_t.decode_split_kv == 2
    # S_q 4 at 32 / 2 (64 rows) rides the 64-column tile in ONE unit under both policies; the cap restores the former pair (prefill /
    # two units of the 32-column tile), the claimed MEASURED form under token_units
    for policy, want in ((DECODE_ROUTE_SINGLE_UNIT, (_T, 64, 1, 1, None)), (DECODE_ROUTE_TOKEN_UNITS, (_T, 64, 1, 1, None))):
        b, _ = _make_block(16, 4, 4, decode_route_policy=policy)
        rr = b.decode_route
        assert (rr.tile, rr.n_q, rr.token_units, rr.kv_streams, rr.q_tile_max) == want and not rr.claimed and b._sdpa.paged_dense, (policy, rr)
    cap_s, _ = _make_block(16, 32, 4, decode_q_tile_max=32)
    assert cap_s.decode_q_tile_max == 32 and cap_s.decode_route.prefill_tile and cap_s.decode_route.q_tile_max == 32 and cap_s._sdpa.thd
    cap_b, _ = _make_block(16, 4, 4, decode_q_tile_max=32)  # below the fill the capped route is two units of the 32-column tile
    assert (cap_b.decode_route.tile, cap_b.decode_route.n_q, cap_b.decode_route.token_units, cap_b.decode_route.q_tile_max) == (
        _T,
        32,
        2,
        32,
    ) and cap_b._sdpa.paged_dense
    cap_t, _ = _make_block(16, 4, 4, decode_route_policy=DECODE_ROUTE_TOKEN_UNITS, decode_q_tile_max=32, decode_split_kv=2)
    rc = cap_t.decode_route
    assert (rc.tile, rc.n_q, rc.token_units, rc.kv_streams, rc.claimed, rc.q_tile_max) == (_T, 32, 2, 2, True, 32) and cap_t._sdpa.paged_dense, rc
    # a group the prefill tile cannot pack: both policies coincide at S_q 4 (one 64-column unit) and S_q 8 (two units of it)
    for s_q, want in ((4, (64, 1, True, 1)), (8, (64, 2, True, 2))):
        for policy in DECODE_ROUTE_POLICIES:
            b, _ = _make_block(12, 2, s_q, decode_route_policy=policy)
            rr = b.decode_route
            assert rr.decode_tile and (rr.n_q, rr.token_units, rr.pack_gqa, rr.kv_streams) == want and b._sdpa.paged_dense, (s_q, policy, rr)
    with pytest.raises(ValueError, match="decode_route_policy must be one of"):
        _make_block(16, 4, 4, decode_route_policy="fastest")
    with pytest.raises(ValueError, match="decode_q_tile_max must be one of"):
        _make_block(16, 4, 4, decode_q_tile_max=48)
    for extra in ({"decode_route_policy": "token_units"}, {"decode_q_tile_max": 32}):
        with pytest.raises(ValueError, match="belong to the EXTEND form or the DECODE form"):
            _make_block(
                16,
                4,
                4,
                kv_source="self",
                paged_kv_page_size=0,
                sample_k_cache=None,
                sample_v_cache=None,
                sample_block_table=None,
                **extra,
            )
    # the standalone stage: a dense paged stage cannot bind the prefill route (the block builds the THD stage for it)
    from cudnn.api_base import TensorDesc

    geom, _ = _geom(16)
    dev = torch.device(_DEV)
    kc = torch.empty(4, 2, 16, _D, dtype=torch.bfloat16, device=_DEV)
    kd = TensorDesc(dtype=kc.dtype, shape=tuple(kc.shape), stride=tuple(kc.stride()), stride_order=(3, 2, 1, 0), device=kc.device)
    bd = TensorDesc(dtype=torch.int32, shape=(1, 4), stride=(4, 1), stride_order=(1, 0), device=dev)
    paged = dict(kv_source="paged", paged_page_size=16, k_cache_desc=kd, v_cache_desc=kd, block_table_desc=bd)
    bd32 = TensorDesc(dtype=torch.int32, shape=(32, 4), stride=(4, 1), stride_order=(1, 0), device=dev)
    paged32 = dict(paged, block_table_desc=bd32)
    with pytest.raises(NotImplementedError, match="prefill tile.*thd=True"):
        _Sdpa(geom, batch=32, seq_len=5, dtype=torch.bfloat16, device=dev, want_lse=False, **paged32)
    one = _Sdpa(geom, batch=1, seq_len=5, dtype=torch.bfloat16, device=dev, want_lse=False, **paged)  # below the fill: the decode tile, dense
    assert one.paged_dense and one.decode_route.decode_tile and one.decode_route.token_units == 2 and one.decode_route.batch == 1
    st = _Sdpa(
        geom,
        batch=32,
        seq_len=4,
        dtype=torch.bfloat16,
        device=dev,
        want_lse=False,
        thd=True,
        num_sequences=32,
        max_seq_len=4,
        route=decode_route(geom, 4, max_q_tile=32, batch=32),
        **paged32,
    )
    assert st.thd and st.decode_route.prefill_tile and not st.paged_dense
    with pytest.raises(ValueError, match="prefill route"):
        _Sdpa(
            geom,
            batch=1,
            seq_len=4,
            dtype=torch.bfloat16,
            device=dev,
            want_lse=False,
            thd=True,
            num_sequences=1,
            max_seq_len=4,
            route=decode_route(geom, 4, policy=DECODE_ROUTE_TOKEN_UNITS),
            **paged,
        )


def test_the_prefill_routes_q_lengths_are_a_workspace_slot_not_plan_owned_memory():
    """The paged THD leg's per-sequence Q lengths on the prefill route (the constant ``S_q`` of every sequence) are a ``[B]`` int32 slot
    of the caller's workspace (Rule 8 recipes R2 + R4), reserved by the carve at the very END and only on that route: the layout with
    ``decode_q_lens=0`` is byte-identical to before, a block on the decode tile reserves none, a block on the prefill tile reserves it
    at declaration (before ``compile()``); the block holds no plan-owned device tensor for them and no first-execute flag -- the fill
    rides every execute."""
    import dataclasses

    from cudnn.gated_attention_block.api import _WS_ALIGN, _align_up, _plan_workspace

    geom, _ = _geom(16)
    base = _plan_workspace(geom, 4, 8, torch.bfloat16, False, False, True)
    assert base == _plan_workspace(geom, 4, 8, torch.bfloat16, False, False, True, decode_q_lens=0) and base.decode_q_lens == -1
    lay = _plan_workspace(geom, 4, 8, torch.bfloat16, False, False, True, decode_q_lens=4)
    assert lay.decode_q_lens == base.total_bytes and lay.decode_q_lens % _WS_ALIGN == 0, "the slot is appended, aligned"
    assert lay.total_bytes == lay.engine_scratch == base.total_bytes + _align_up(4 * 4)
    assert dataclasses.replace(lay, decode_q_lens=-1, total_bytes=base.total_bytes, engine_scratch=base.engine_scratch) == base
    on, _ = _make_block(16, 32, 8)  # 32 / 2 at S_q 8 with 32 sequences (64 CTAs: the fill): the prefill tile
    off, _ = _make_block(16, 32, 2)  # 32 / 2 at S_q 2: the decode tile
    below, _ = _make_block(16, 4, 8)  # 32 / 2 at S_q 8 below the fill: two units of the decode tile, no slot
    assert on.decode_route.prefill_tile and off.decode_route.decode_tile and below.decode_route.decode_tile
    lay_on, lay_off = on._layout(), off._layout()
    assert lay_on.decode_q_lens == lay_on.total_bytes - _align_up(32 * 4) >= 0 and lay_off.decode_q_lens == -1 and below._layout().decode_q_lens == -1
    for blk in (on, off):
        assert not any(k.startswith("_decode_q_lens") for k in vars(blk)), "no plan-owned Q-length tensor, no write-once flag"
    assert callable(getattr(on, "_fill_decode_q_lens", None))


def test_the_envelope_names_the_policy_and_the_claimed_bound(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom, _ = _geom(16)
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
    d = env.decode
    assert d.served and d.policy == DEFAULT_DECODE_ROUTE_POLICY and d.claimed_max_s_q == DECODE_CLAIMED_MAX_S_Q and d.max_s_q == DECODE_MAX_S_Q
    assert d.route(2).claimed and d.route(2).decode_tile and d.route(4).decode_tile and d.route(4).n_q == 64 and not d.route(4).claimed
    assert d.route(5).decode_tile and d.route(5).token_units == 2 and d.route(8).token_units == 2, "no batch: the small-batch reading"
    assert d.route(5, batch=32).prefill_tile and d.route(8, batch=32).prefill_tile and d.q_tile_max is None and d.route(4).q_tile_max is None
    assert d.route(8, batch=31).decode_tile and d.route(8, batch=32).batch == 32 and d.route(8).batch is None
    assert d.route(9).served and not d.route(9).pack_gqa and d.route(9).kv_streams == 16 and d.route(32).served and not d.route(33).served
    assert "decode_rows" in d.declines and "decode_s_q" not in d.declines and d.max_s_q == d.max_rows == DECODE_MAX_ROWS == 32
    env_t = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy=DECODE_ROUTE_TOKEN_UNITS)
    assert env_t.decode.policy == DECODE_ROUTE_TOKEN_UNITS and env_t.decode.route(5).token_units == 2 and not env_t.decode.route(5).claimed
    env_c = GatedAttentionBlockFwd.serving_envelope(
        geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy=DECODE_ROUTE_TOKEN_UNITS, decode_q_tile_max=32
    )
    assert env_c.decode.q_tile_max == 32 and (env_c.decode.route(4).n_q, env_c.decode.route(4).token_units, env_c.decode.route(4).claimed) == (32, 2, True)
    assert all(x.q_tile_max == 32 for x in env_c.decode.routes) and env_c.decode.route(8).n_q == 16 and not env_c.decode.route(8).pack_gqa
    with pytest.raises(ValueError, match="decode_q_tile_max must be one of"):
        GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_q_tile_max=48)
    assert "DECODE_MAX_ROWS = 32" in " ".join(d.requires) and "S_q <= 4" in " ".join(d.requires) and "S_q = 8" in " ".join(d.requires)
    with pytest.raises(ValueError, match="decode_route_policy must be one of"):
        GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy="fastest")


# ---------------------------------------------------------------------------
# Rubin: the cells
# ---------------------------------------------------------------------------

_LENS = (100, 384, 500, 0)  # one KV tile with a tail, the ring depth, four tiles with a tail, a PADDED entry (kv_lens > S_q on every live row)
_LENS_FILL = _LENS * 8  # 32 sequences x 2 KV heads = 64 CTAs: the prefill route's fill at 32 / 2 (the standard cell's lengths, repeated)
_CASES = {}


def _case(key, *args, **kw):
    if key not in _CASES:
        _CASES[key] = _Case(*args, **kw).compile()
    return _CASES[key]


def _readback(case):
    """What the adapter COMPILED for stage (4): the template, the body file, the packing / THD / paged / gate flags of its config."""
    impl = case.blk._sdpa._impl
    mod = impl._k_mod
    cfg = getattr(mod, "CFG", None)
    return dict(
        template=impl.kernel_template,
        file=mod.__file__,
        heads_per_tile=getattr(mod, "HEADS_PER_TILE", None),
        pack_gqa=int(getattr(cfg, "PACK_GQA", 0)),
        thd=int(getattr(cfg, "THD_VARLEN", 0)),
        paged=int(getattr(cfg, "PAGED_KV", 0)),
        epilogue_gate=int(getattr(cfg, "EPILOGUE_GATE", 0)) if hasattr(cfg, "EPILOGUE_GATE") else getattr(getattr(mod, "PARAMS", None), "epilogue_gate", None),
        n_q=getattr(cfg, "N_Q", None),
        adapter_pack=impl.pack_gqa,
        adapter_split=impl.split_kv,
    )


def _pin_route(case):
    """The compiled kernel against the declared route: the prefill tile = the shared SM100 d256 paged THD body with the WHOLE group packed
    and the gate in its epilogue; the decode tile = the Rubin swap-AB tile at the route's width, the gate in the combine."""
    r, rb = case.blk.decode_route, _readback(case)
    assert rb["template"] == r.tile, (r, rb)
    if r.prefill_tile:
        assert rb["file"].endswith("sm100/prefill_d256_f16.py") and rb["heads_per_tile"] == r.group and rb["adapter_pack"] is True, rb
        assert (rb["pack_gqa"], rb["thd"], rb["paged"], rb["adapter_split"]) == (1, 1, 1, 1), rb
        assert bool(rb["epilogue_gate"]) is case.blk.fuse_gate and case.blk.gate_fused is case.blk.fuse_gate and case.blk.decode_split_kv == 1, rb
    else:
        assert rb["file"].endswith("sm107/decode_d256_f16.py") and rb["n_q"] == r.n_q and rb["heads_per_tile"] == (r.group if r.pack_gqa else 1), rb
        assert (rb["adapter_pack"] is True) == r.pack_gqa and rb["adapter_split"] == case.blk.decode_split_kv, rb
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0), "block 0 (the NULL block) must never be written"
    return r, rb


# (group, S_q): the MTP cells at the three geometries on their routes at the standard cell's four sequences -- below the prefill
# route's fill, so TWO units of the 64-column tile at 32 / 2 S_q 5 / 8 (80 / 128 packed rows), 8 / 1 S_q 9 / 16 (72 / 128) and 4 / 1 S_q
# 17 / 32 (68 / 128; the prefill route's cells at the fill are the test below); the 64-column tile in ONE unit at 32 / 2 S_q 4 (64 rows),
# 8 / 1 S_q 5 / 8 (40 / 64) and 24 / 2 S_q 4 (48); in TWO units at 24 / 2 S_q 8 (96 rows: the prefill tile cannot pack a 12-head group,
# and the token-unit arm replaced the unpacked 12-stream arm); the UNPACKED arm at 32 / 2 S_q 12 (the 16-column tile, one head per CTA,
# 16 streams) and S_q 32 (the 32-column tile: the widest step the form serves).
_ORACLE_CELLS = [(16, 4), (16, 5), (16, 8), (8, 5), (8, 8), (8, 9), (8, 16), (4, 17), (4, 32), (12, 4), (12, 8), (16, 12), (16, 32)]


@requires_rubin
@pytest.mark.parametrize("group, s_q", _ORACLE_CELLS, ids=[f"{_GROUPS[g][0]}_{_GROUPS[g][1]}_sq{s}" for g, s in _ORACLE_CELLS])
def test_the_mtp_step_matches_the_oracle_on_its_route(group, s_q):
    """Every MTP cell against the per-sequence fp32 oracle under the dense suite's budget (cos > 0.999 on ``out``), the padded entry
    EXACTLY 0 with block 0 untouched, the compiled kernel pinned against the declared route (the shared d256 paged THD body with the
    group packed and the gate in its epilogue on the prefill route; the swap-AB tile otherwise)."""
    case = _case(("std", group, s_q), group, s_q, _LENS)
    case.run()
    case.check_against_oracle()
    r, _ = _pin_route(case)
    want = _SINGLE_UNIT[group][s_q]
    assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == want[:5] and r.batch == len(_LENS), (r, want)


@requires_rubin
@pytest.mark.parametrize("s_q", [5, 8], ids=["sq5", "sq8"])
def test_the_prefill_route_matches_the_oracle_at_the_fill(s_q):
    """32 / 2 at ``S_q`` 5 / 8 with 32 sequences (64 CTAs: the prefill route's fill, ``DECODE_PREFILL_ROUTE_MIN_CTAS``): the paged PREFILL
    tile with the whole group packed and the gate in its epilogue, against the per-sequence fp32 oracle, the padded entries EXACTLY 0,
    the compiled kernel pinned against the declared route and the route's batch read back."""
    case = _case(("fill", 16, s_q), 16, s_q, _LENS_FILL)
    case.run()
    case.check_against_oracle()
    r, _ = _pin_route(case)
    assert r.prefill_tile and (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == _FILLED[16][s_q][:5] and r.batch == 32, r


@requires_rubin
def test_the_prefill_route_and_the_two_unit_route_agree_on_identical_inputs():
    """The A/B's two arms at 32 / 2, ``S_q = 5`` (the first step past the single-unit table at this group) on IDENTICAL inputs (one seed:
    the same weights, inputs, prefix rows, pages and slots) at the prefill route's fill (32 sequences): the prefill route (one KV stream,
    the gate in the body) and the two-unit decode tile (two units of the 64-column width, two streams, the gate in the split combine)
    both within the oracle's budget and within one another's -- two kernels, two summation orders, so not bitwise; cos > 0.9999 between
    them and the max |diff| printed."""
    a = _case(("ab", 16, 5, "prefill"), 16, 5, _LENS_FILL, seed=31)
    b = _case(("ab", 16, 5, "two_unit"), 16, 5, _LENS_FILL, seed=31, blk_kw=dict(decode_route_policy=DECODE_ROUTE_TOKEN_UNITS))
    assert (
        torch.equal(a.h, b.h)
        and torch.equal(a.w["w_qkvg"], b.w["w_qkvg"])
        and torch.equal(a.block_table, b.block_table)
        and torch.equal(a.slot_mapping, b.slot_mapping)
    )
    assert a.blk.decode_route.prefill_tile and b.blk.decode_route.decode_tile and b.blk.decode_route.token_units == 2
    out_a, out_b = a.run().clone(), b.run().clone()
    a.check_against_oracle(out_a, what="prefill route")
    b.check_against_oracle(out_b, what="two-unit route")
    _pin_route(a)
    _pin_route(b)
    live = a.live
    c = _cos(out_a[live], out_b[live])
    md = (out_a[live].float() - out_b[live].float()).abs().max().item()
    print(f"[32/2 S_q 5] prefill route vs two-unit route: cos {c:.6f} max|diff| {md:.3e}")
    assert c > 0.9999 and torch.equal(out_a[~live], out_b[~live]) and (out_a[~live] == 0).all()


@requires_rubin
def test_the_prefill_route_gate_fused_vs_the_separate_gate_launch():
    """``fuse_gate`` on the prefill route rides the paged body's epilogue (one launch fewer); against the same block with the gate as its own
    launch: both within the oracle's budget, ``out`` cos > 0.9999 between them."""
    fused = _case(("gate", 16, 5, "fused"), 16, 5, _LENS_FILL, seed=37)
    plain = _case(("gate", 16, 5, "launch"), 16, 5, _LENS_FILL, seed=37, blk_kw=dict(fuse_gate=False))
    assert fused.blk.decode_route.prefill_tile and plain.blk.decode_route.prefill_tile
    assert fused.blk.gate_fused and fused.blk._gate is None and not plain.blk.gate_fused and plain.blk._gate is not None
    a, b = fused.run().clone(), plain.run().clone()
    fused.check_against_oracle(a, what="gate in the body")
    plain.check_against_oracle(b, what="gate launch")
    assert _readback(fused)["epilogue_gate"] == 1 and _readback(plain)["epilogue_gate"] == 0
    assert _cos(a[fused.live], b[fused.live]) > 0.9999


@requires_rubin
def test_two_launches_are_bitwise_and_the_workspace_is_honest_on_the_prefill_route():
    case = _case(("fill", 16, 8), 16, 8, _LENS_FILL)
    assert case.blk.decode_route.prefill_tile
    a = case.run().clone()
    b = case.run().clone()
    assert torch.equal(a, b)
    blk = case.blk
    need = blk.get_workspace_size()
    assert blk._sdpa.scratch_workspace_bytes() > 0 and need >= blk._layout().total_bytes + blk._sdpa.scratch_workspace_bytes()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    out = torch.empty_like(case.out)
    case.execute(out=out, ws=ws)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.check_against_oracle(out, what="sentinel-tailed workspace")


@requires_rubin
def test_capture_of_the_first_execute_at_s_q_eight_replays_bitwise_and_takes_new_lengths_tables_slots_and_inputs():
    """The serving stack's defaults at the MTP-7 verify step (32 / 2, ``S_q = 8``, the prefill route): the FIRST execute of a fresh
    declaration captured on a NEW stream with no eager warm-up under ``set_sync_debug_mode("error")``; the capture launches nothing,
    the replay is BITWISE a fresh eager run; a padded batch replays correctly; new ``kv_lens`` / ``block_table`` / ``slot_mapping`` /
    ``h`` through the captured tensors replay bitwise the eager run over them."""
    lens = (300, 0, 1000, 77) * 8  # 32 sequences: the prefill route's fill
    case = _Case(16, 8, lens, seed=41)  # fresh: never executed
    assert case.blk.decode_route.prefill_tile and case.blk.decode_route.batch == 32
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
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(case.out, replayed), "the replay of the captured first execute differs from the eager run"
        sib = _Case(16, 8, (301, 0, 1000, 77) * 8, seed=43, table_row_stride=case.block_table.shape[1])
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
def test_an_eager_execute_after_a_captured_first_execute_with_no_replay_reads_its_own_q_lengths():
    """The prefill route fills the paged THD leg's Q lengths on EVERY execute (a workspace slot, Rule 8 recipes R2 + R4): the FIRST
    execute of a fresh declaration is CAPTURED (nothing runs) and the block is then run EAGERLY before any replay -- the eager run
    matches the oracle (a write-once flag flipped by the capture would have handed the leg an unwritten vector), the slot reads ``S_q``
    on every row afterwards, a fresh workspace whose slot is poisoned is served the same way (the fill rides the execute, not the plan),
    and the later replay of the captured first execute is bitwise the eager run."""
    lens = (300, 0, 1000, 77) * 8  # 32 sequences: the prefill route's fill
    case = _Case(16, 8, lens, seed=47)  # fresh: never executed
    assert case.blk.decode_route.prefill_tile and case.blk.decode_route.batch == 32
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.empty(case.blk.get_workspace_size(), dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            case.execute()  # the FIRST execute of this declaration, captured: nothing runs
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all(), "the capture launched work"
        case.execute()  # EAGER, before any replay
        torch.cuda.synchronize()
        eager = case.out.clone()
        case.check_against_oracle(eager, what="eager execute after a captured first execute, no replay")
        lay = case.blk._layout()
        slot = slice(lay.decode_q_lens, lay.decode_q_lens + 4 * len(lens))
        assert case.ws[slot].view(torch.int32).tolist() == [8] * len(lens)
        ws2 = torch.empty_like(case.ws)
        ws2[slot].fill_(0xFF)  # the slot poisoned (-1 per row): the execute must fill it itself
        out2 = torch.empty_like(case.out)
        case.execute(out=out2, ws=ws2)
        torch.cuda.synchronize()
        assert ws2[slot].view(torch.int32).tolist() == [8] * len(lens) and torch.equal(out2, eager)
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(case.out, eager), "the replay of the captured first execute differs from the eager run"
    finally:
        graph.reset()
