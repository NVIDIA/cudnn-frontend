# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's DECODE route for the speculative (MTP) verify step -- table-driven, one KV stream per sequence
where a kernel offers it, the declared envelope ``S_q <= 8``.

The route (``cudnn.gated_attention_block.decode_route``) reads ONE table, ``config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES`` --
the Rubin decode tile's widths that take a step in ONE token unit (the KV streamed once per sequence and KV head) -- and applies
ONE rule under the default policy (``decode_route_policy="single_unit"``): packed rows ``S_q x group`` within the widest width
ride that width in one unit; past the table the step rides the paged d256 PREFILL tile with the whole group packed (one Q tile per
sequence and KV head: the KV streamed once) wherever that tile packs the group (a group dividing its 128 rows); a group the
prefill tile cannot pack (12 = 24 / 2, 6 = 24 / 4) keeps the decode tile's token-unit arm -- up to two units, two streams, the
measured MTP form -- and the unpacked tile (one head per CTA) past it, the fewest streams any kernel offers it today.  The policy
``"token_units"`` is the SDPA adapter's own route verbatim (the widest tile in up to two token units, then the unpacked tile), the
A/B's other arm.  Past ``S_q = 8`` (``DECODE_MAX_S_Q``) or 128 packed rows (``DECODE_MAX_ROWS``) the step is a typed decline.

Host cells (any device): the table and the rule, a wider width slotted into the table without touching the rule, the adapter's
route unchanged by the re-expression (every ``(S_q, group)`` against the shared SM100 rule with the Rubin constants), the route
per policy at the three serving geometries of the MTP cells (24 / 2, 32 / 2, 8 / 1) and the corner groups, the two reject rows,
the declaration facts on the prefill route (the THD stage, the split, the gate in the paged body's epilogue), the envelope per
policy.  Rubin cells: the prefill route vs the per-sequence fp32 oracle at ``S_q`` 4 / 5 / 8 with a padded entry, the prefill route
vs the two-unit route on IDENTICAL inputs at ``S_q = 4``, the unpacked arm at 24 / 2 ``S_q = 8``, two launches bitwise, the
workspace honest, CUDA-graph capture of the FIRST execute at ``S_q = 8`` and its replay over new lengths / tables / slots / inputs.
"""

import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    DECODE_CLAIMED_MAX_S_Q,
    DECODE_MAX_ROWS,
    DECODE_MAX_S_Q,
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

# (tile, n_q, token_units, pack_gqa, kv_streams, claimed) per (group, S_q) under the DEFAULT policy ("single_unit") -- the Rubin
# table (16, 32) and the prefill tile packing the groups that divide 128.  Group 12 (24 / 2) and 6 (24 / 4) cannot pack on the
# prefill tile, so past the table they keep the token-unit arm (two units) and the unpacked tile.
_P, _T = PREFILL_TILE, DECODE_TILE
_SINGLE_UNIT = {
    16: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 32, 1, True, 1, True),
        3: (_P, 0, 0, True, 1, False),
        4: (_P, 0, 0, True, 1, False),
        5: (_P, 0, 0, True, 1, False),
        8: (_P, 0, 0, True, 1, False),
        9: (None, 0, 0, False, 0, False),
    },
    12: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 32, 1, True, 1, True),
        3: (_T, 32, 2, True, 2, True),
        4: (_T, 32, 2, True, 2, True),
        5: (_T, 16, 1, False, 12, False),
        8: (_T, 16, 1, False, 12, False),
        9: (None, 0, 0, False, 0, False),
    },
    8: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 16, 1, True, 1, True),
        3: (_T, 32, 1, True, 1, True),
        4: (_T, 32, 1, True, 1, True),
        5: (_P, 0, 0, True, 1, False),
        8: (_P, 0, 0, True, 1, False),
        9: (None, 0, 0, False, 0, False),
    },
    6: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 16, 1, True, 1, True),
        4: (_T, 32, 1, True, 1, True),
        5: (_T, 32, 1, True, 1, False),
        6: (_T, 32, 2, True, 2, False),
        8: (_T, 32, 2, True, 2, False),
        9: (None, 0, 0, False, 0, False),
    },
    4: {
        1: (_T, 16, 1, True, 1, True),
        4: (_T, 16, 1, True, 1, True),
        5: (_T, 32, 1, True, 1, False),
        8: (_T, 32, 1, True, 1, False),
        9: (None, 0, 0, False, 0, False),
    },
}
# The same cells under "token_units" (the adapter's route verbatim, inside the declared envelope).
_TOKEN_UNITS = {
    16: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 32, 1, True, 1, True),
        3: (_T, 32, 2, True, 2, True),
        4: (_T, 32, 2, True, 2, True),
        5: (_T, 16, 1, False, 16, False),
        8: (_T, 16, 1, False, 16, False),
        9: (None, 0, 0, False, 0, False),
    },
    12: _SINGLE_UNIT[12],
    8: {
        1: (_T, 16, 1, True, 1, True),
        2: (_T, 16, 1, True, 1, True),
        3: (_T, 32, 1, True, 1, True),
        4: (_T, 32, 1, True, 1, True),
        5: (_T, 32, 2, True, 2, False),
        8: (_T, 32, 2, True, 2, False),
        9: (None, 0, 0, False, 0, False),
    },
    6: _SINGLE_UNIT[6],
    4: _SINGLE_UNIT[4],
}
_TABLES = {DECODE_ROUTE_SINGLE_UNIT: _SINGLE_UNIT, DECODE_ROUTE_TOKEN_UNITS: _TOKEN_UNITS}


def _check_route(r: DecodeRoute, want, *, group, s_q, policy):
    tile, n_q, units, packed, streams, claimed = want
    assert isinstance(r, DecodeRoute) and (r.s_q, r.group, r.policy) == (s_q, group, policy), r
    assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams, r.claimed) == want, (group, s_q, policy, r)
    assert r.served is (tile is not None) and r.decode_tile is (tile == _T) and r.prefill_tile is (tile == _P), r
    assert (r.reason == "") is r.served and (r.served or "EXTEND form" in r.reason), r
    if tile == _T and packed:
        assert units == -(-s_q // (n_q // group)) and streams == units, r
    elif tile == _T:
        assert units == 1 and s_q <= n_q and streams == (1 if group == 1 else group), r
    elif tile == _P:
        assert units == 0 and streams == 1 and packed and s_q * group <= DECODE_MAX_ROWS and 128 % group == 0, r


# ---------------------------------------------------------------------------
# The table, the rule, the adapter's route
# ---------------------------------------------------------------------------


def test_the_single_unit_table_and_the_rule():
    """ONE ordered table of single-unit widths; THE rule = the narrowest width holding the packed rows, else 0 (past the table);
    the routed maximum is the table's widest entry and never exceeds the width the tile compiles for."""
    from cudnn.sdpa.fwd import config_sm100, config_sm107

    table = config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES
    assert table == (16, 32) and list(table) == sorted(table) and all(w > 0 for w in table)
    assert config_sm107.D256_DECODE_ROUTED_MAX_Q_ROWS == table[-1] <= config_sm100.D256_DECODE_MAX_Q_ROWS
    assert config_sm107.D256_DECODE_ROUTED_MAX_TOKEN_UNITS == 2  # the adapter's token-unit cap, unchanged
    rule = config_sm107.decode_d256_single_unit_q_tile
    for s_q in range(0, 70):
        for g in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 33, 64):
            rows = s_q * g
            want = 0 if (s_q <= 0 or g <= 0) else next((w for w in table if rows <= w), 0)
            assert rule(s_q, g) == want, (s_q, g, rule(s_q, g), want)
    assert rule(1, 12) == 16 and rule(2, 12) == 32 and rule(3, 12) == 0 and rule(4, 12) == 0 and rule(1, 32) == 32 and rule(2, 32) == 0
    assert rule(32, 1) == 32 and rule(33, 1) == 0 and rule(16, 1) == 16


def test_a_wider_width_slots_into_the_table_without_touching_the_rule(monkeypatch):
    """A 64-column entry (another lane's tile) appended to the table routes the 24 / 2 step at ``S_q = 4`` -- 48 rows -- in ONE unit
    of it, through the unchanged rule and the unchanged adapter route; the block's route follows."""
    from cudnn.sdpa.fwd import config_sm107

    monkeypatch.setattr(config_sm107, "D256_DECODE_SINGLE_UNIT_Q_TILES", (16, 32, 64))
    monkeypatch.setattr(config_sm107, "D256_DECODE_ROUTED_MAX_Q_ROWS", 64)
    assert config_sm107.decode_d256_single_unit_q_tile(4, 12) == 64 and config_sm107.decode_d256_single_unit_q_tile(5, 12) == 64
    assert config_sm107.decode_d256_single_unit_q_tile(6, 12) == 0 and config_sm107.decode_d256_single_unit_q_tile(2, 12) == 32
    assert config_sm107.decode_d256_q_tile(4, 12) == 64 and config_sm107.decode_d256_q_tile(4, 16) == 64  # one unit of the new width
    assert config_sm107.decode_d256_q_tile(8, 12) == 64  # 96 rows: two units of the WIDEST width (the adapter's token-unit arm)
    assert config_sm107.decode_d256_q_tile(11, 12) == 0  # three units: past the adapter's cap
    geom, _ = _geom(12)
    r = decode_route(geom, 4)
    assert (r.tile, r.n_q, r.token_units, r.kv_streams) == (DECODE_TILE, 64, 1, 1), r


def test_the_adapters_route_is_the_former_rule_verbatim():
    """``config_sm107.decode_d256_q_tile`` re-expressed on the table equals the shared SM100 rule with the Rubin constants at every
    ``(S_q, group)``: the standalone adapter's placement (api_dsl / engines / heuristics) does not move.  ``max_token_units=1``
    is the single-unit rule alone."""
    from cudnn.sdpa.fwd import config_sm100, config_sm107

    for s_q in range(0, 70):
        for g in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 33, 64):
            old = config_sm100.decode_d256_q_tile(s_q, g, config_sm107.D256_DECODE_ROUTED_MAX_Q_ROWS, 2 if g > 1 else 1)
            assert config_sm107.decode_d256_q_tile(s_q, g) == old, (s_q, g, config_sm107.decode_d256_q_tile(s_q, g), old)
            assert config_sm107.decode_d256_q_tile(s_q, g, max_token_units=1) == config_sm107.decode_d256_single_unit_q_tile(s_q, g), (s_q, g)
    assert config_sm107.decode_d256_q_tile(4, 12) == 32 and config_sm107.decode_d256_q_tile(5, 12) == 0 and config_sm107.decode_d256_q_tile(4, 16) == 32


# ---------------------------------------------------------------------------
# The route per policy, the envelope, the reject cells
# ---------------------------------------------------------------------------


def test_constants_and_policies():
    assert (DECODE_CLAIMED_MAX_S_Q, DECODE_MAX_S_Q, DECODE_MAX_ROWS) == (4, 8, 128)
    assert DECODE_ROUTE_POLICIES == (DECODE_ROUTE_SINGLE_UNIT, DECODE_ROUTE_TOKEN_UNITS) == ("single_unit", "token_units")
    assert DEFAULT_DECODE_ROUTE_POLICY == DECODE_ROUTE_SINGLE_UNIT
    assert (DECODE_TILE, PREFILL_TILE) == ("decode_d256_f16", "prefill_d256_f16")
    from cudnn.sdpa.fwd.config_sm100 import CfgD256

    assert DECODE_MAX_ROWS == CfgD256.TILE_M  # the prefill tile's Q rows: one packed Q tile per (sequence, KV head)


@pytest.mark.parametrize("policy", DECODE_ROUTE_POLICIES)
@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_the_route_per_policy_at_every_served_group(group, policy):
    """Which tile each ``S_q`` rides, per policy: the single-unit table, the packed prefill tile where the group divides its rows, the
    token-unit / unpacked arms where it does not (and everywhere under ``token_units``); ``kv_streams`` names the KV streams per
    (sequence, KV head); ``claimed`` is the packed DECODE tile at ``S_q <= 4`` only; the default policy is ``single_unit``."""
    geom, _ = _geom(group)
    for s_q, want in _TABLES[policy][group].items():
        _check_route(decode_route(geom, s_q, policy=policy), want, group=group, s_q=s_q, policy=policy)
        if policy == DEFAULT_DECODE_ROUTE_POLICY:
            assert decode_route(geom, s_q) == decode_route(geom, s_q, policy=policy)
    for s_q in range(1, DECODE_MAX_S_Q + 1):
        r = decode_route(geom, s_q, policy=policy)
        assert r.served and r.kv_streams >= 1 and (r.decode_tile or r.prefill_tile), r
        if policy == DECODE_ROUTE_SINGLE_UNIT and 128 % group == 0:
            assert r.kv_streams == 1, ("a group the prefill tile packs streams the KV once at every S_q", r)
        if policy == DECODE_ROUTE_SINGLE_UNIT:
            assert r.kv_streams <= decode_route(geom, s_q, policy=DECODE_ROUTE_TOKEN_UNITS).kv_streams, ("never more streams than the adapter's route", r)


def test_the_corner_groups():
    """MHA stays the unpacked 16-column tile (one stream: one head per KV head); group 32 (64 / 2) packs on the 32-column tile at ``S_q = 1``
    and on the prefill tile up to 128 rows; group 64 rides the prefill tile from ``S_q = 1``; an odd group (3) packs on the decode tile
    at every ``S_q <= 8`` (whole-group packing has no divisibility rule there)."""
    mk = lambda h_q, h_kv: GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=h_q, h_kv=h_kv, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)  # noqa: E731
    mha = mk(4, 4)
    for s_q in range(1, DECODE_MAX_S_Q + 1):
        _check_route(decode_route(mha, s_q), (_T, 16, 1, False, 1, False), group=1, s_q=s_q, policy=DECODE_ROUTE_SINGLE_UNIT)
    g32 = mk(64, 2)
    _check_route(decode_route(g32, 1), (_T, 32, 1, True, 1, True), group=32, s_q=1, policy=DECODE_ROUTE_SINGLE_UNIT)
    for s_q in (2, 3, 4):
        _check_route(decode_route(g32, s_q), (_P, 0, 0, True, 1, False), group=32, s_q=s_q, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g32, 2, policy=DECODE_ROUTE_TOKEN_UNITS), (_T, 32, 2, True, 2, True), group=32, s_q=2, policy=DECODE_ROUTE_TOKEN_UNITS)
    _check_route(decode_route(g32, 3, policy=DECODE_ROUTE_TOKEN_UNITS), (_T, 16, 1, False, 32, False), group=32, s_q=3, policy=DECODE_ROUTE_TOKEN_UNITS)
    r = decode_route(g32, 5)  # 160 packed rows: past the prefill tile's 128
    assert not r.served and r.tile is None and "128" in r.reason and "EXTEND form" in r.reason, r
    g64 = mk(128, 2)
    _check_route(decode_route(g64, 1), (_P, 0, 0, True, 1, False), group=64, s_q=1, policy=DECODE_ROUTE_SINGLE_UNIT)
    _check_route(decode_route(g64, 2), (_P, 0, 0, True, 1, False), group=64, s_q=2, policy=DECODE_ROUTE_SINGLE_UNIT)
    assert not decode_route(g64, 3).served
    g3 = mk(12, 4)
    for s_q in range(1, DECODE_MAX_S_Q + 1):
        r = decode_route(g3, s_q)
        assert r.decode_tile and r.pack_gqa and r.token_units == 1 and r.n_q == (16 if 3 * s_q <= 16 else 32), r
    d128 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=128, rope_dim=_ROPE, causal_bottom_right=True)
    assert not decode_route(d128, 1).served and "d_head=128" in decode_route(d128, 1).reason
    with pytest.raises(ValueError, match="decode_route_policy must be one of"):
        decode_route(mha, 1, policy="fastest")


def test_the_reject_cells_s_q_nine_and_rows_past_128(monkeypatch):
    """Past the DECLARED envelope the route says not served with the reason, the declaration raises it (typed, by name) and the envelope
    predicts it -- under both policies: ``S_q = 9`` at every served group (the ``decode_s_q`` row), 160 packed rows (the ``decode_rows``
    row) -- while ``S_q = 8`` and 128 rows are served."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    for group in sorted(_GROUPS):
        geom, _ = _geom(group)
        for policy in DECODE_ROUTE_POLICIES:
            r = decode_route(geom, DECODE_MAX_S_Q + 1, policy=policy)
            assert (
                not r.served and r.tile is None and r.n_q == 0 and r.kv_streams == 0 and f"S_q <= {DECODE_MAX_S_Q}" in r.reason and "EXTEND form" in r.reason
            ), r
            assert decode_route(geom, DECODE_MAX_S_Q, policy=policy).served
            with pytest.raises(NotImplementedError, match=f"serves S_q <= {DECODE_MAX_S_Q}.*EXTEND form"):
                _make_block(group, 2, DECODE_MAX_S_Q + 1, decode_route_policy=policy)
            env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy=policy)
            assert env.decode.served and env.decode.policy == policy and (env.decode.max_s_q, env.decode.max_rows, env.decode.claimed_max_s_q) == (8, 128, 4)
            assert [x.s_q for x in env.decode.routes] == list(range(1, DECODE_MAX_S_Q + 2)) and not env.decode.routes[-1].served
            assert all(x.policy == policy for x in env.decode.routes) and env.decode.route(DECODE_MAX_S_Q).policy == policy
            assert f"S_q <= {DECODE_MAX_S_Q}" in env.decode.declines["decode_s_q"] and f"{DECODE_MAX_ROWS}" in env.decode.declines["decode_rows"]
        blk8, _ = _make_block(group, 2, DECODE_MAX_S_Q)
        assert blk8.decode_route.served and blk8.decode_route.s_q == DECODE_MAX_S_Q
    # rows past 128: group 32 at S_q 5 = 160 rows, declined by the rows row; S_q 4 = 128 rows served on the prefill tile
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

    with pytest.raises(NotImplementedError, match=f"{DECODE_MAX_ROWS} packed rows.*EXTEND form"):
        declare(5)
    blk = declare(4)
    assert blk.decode_route.prefill_tile and blk.decode_route.kv_streams == 1 and blk.decode_route.s_q * blk.decode_route.group == DECODE_MAX_ROWS
    env = GatedAttentionBlockFwd.serving_envelope(g32, torch.bfloat16, _DEV, device_cc=_SM107)
    assert env.decode.served and not env.decode.route(5).served and env.decode.route(4).prefill_tile


def test_declaration_facts_on_the_prefill_route_and_the_policy_knob():
    """A DECODE block whose route is the prefill tile builds stage (4) as the EXTEND form's binding at the dense step's envelope: a THD
    stage at ``(num_sequences=B, max_seq_len=S_q)`` carrying the route, unsplit, the group packed, the gate in the paged body's epilogue
    (``gate_fused`` with no separate launch); a pinned split there is a typed decline; ``decode_route_policy`` selects the two-unit tile
    instead, is refused off the DECODE form and must be a known policy; the stage list is the serving pipeline."""
    blk, _ = _make_block(16, 4, 4, fuse_gate=True, fuse_norm_rope=True)
    r = blk.decode_route
    assert r.prefill_tile and r.tile == PREFILL_TILE and r.pack_gqa and r.kv_streams == 1 and r.policy == DECODE_ROUTE_SINGLE_UNIT and not r.claimed
    assert blk.decode_route_policy == DECODE_ROUTE_SINGLE_UNIT and blk.decode_split_kv == 1 and blk.seq_lens_present is True
    st = blk._sdpa
    assert isinstance(st, _Sdpa) and st.thd and not st.paged_dense and st.decode_route == r and st.decode_split_kv == 1 and not st.seq_lens_present
    assert (st.num_sequences, st.max_seq_len, st.cu_seqlens, st.batch, st.seq_len) == (4, 4, False, 4, 4)
    assert blk.gate_fused and blk._gate is None and st.fuse_gate, "the gate rides the paged prefill body's epilogue"
    assert [s.name for s in blk._stages] == ["qkv_gate_proj_norm_rope", "sdpa", "out_proj"]
    assert decode_split_choice(r, batch=4, h_kv=2, max_kv_len=1024, sm_count=204, gated=True, h_q=32) == 1
    # a pinned split on the prefill route: typed; 1 is the unsplit prefill body
    with pytest.raises(NotImplementedError, match="prefill route.*unsplit"):
        _make_block(16, 4, 4, decode_split_kv=2)
    blk1, _ = _make_block(16, 4, 4, fuse_gate=True, decode_split_kv=1)
    assert blk1.decode_route.prefill_tile and blk1.decode_split_kv == 1 and blk1.gate_fused
    # the other arm of the A/B: the two-unit decode tile, the gate in the split combine
    blk_t, _ = _make_block(16, 4, 4, fuse_gate=True, fuse_norm_rope=True, decode_route_policy=DECODE_ROUTE_TOKEN_UNITS, decode_split_kv=2)
    rt = blk_t.decode_route
    assert rt.decode_tile and (rt.n_q, rt.token_units, rt.kv_streams, rt.policy, rt.claimed) == (32, 2, 2, DECODE_ROUTE_TOKEN_UNITS, True)
    assert blk_t._sdpa.paged_dense and not blk_t._sdpa.thd and blk_t._sdpa.decode_route == rt and blk_t.gate_fused and blk_t.decode_split_kv == 2
    # a group the prefill tile cannot pack: both policies coincide at S_q 4 (two units) and S_q 8 (the unpacked tile)
    for s_q, want in ((4, (32, 2, True, 2)), (8, (16, 1, False, 12))):
        for policy in DECODE_ROUTE_POLICIES:
            b, _ = _make_block(12, 2, s_q, decode_route_policy=policy)
            rr = b.decode_route
            assert rr.decode_tile and (rr.n_q, rr.token_units, rr.pack_gqa, rr.kv_streams) == want and b._sdpa.paged_dense, (s_q, policy, rr)
    with pytest.raises(ValueError, match="decode_route_policy must be one of"):
        _make_block(16, 4, 4, decode_route_policy="fastest")
    with pytest.raises(ValueError, match="decode_route_policy.*belong to the EXTEND form or the DECODE form"):
        _make_block(
            16,
            4,
            4,
            kv_source="self",
            paged_kv_page_size=0,
            sample_k_cache=None,
            sample_v_cache=None,
            sample_block_table=None,
            decode_route_policy="token_units",
        )
    # the standalone stage: a dense paged stage cannot bind the prefill route (the block builds the THD stage for it)
    from cudnn.api_base import TensorDesc

    geom, _ = _geom(16)
    dev = torch.device(_DEV)
    kc = torch.empty(4, 2, 16, _D, dtype=torch.bfloat16, device=_DEV)
    kd = TensorDesc(dtype=kc.dtype, shape=tuple(kc.shape), stride=tuple(kc.stride()), stride_order=(3, 2, 1, 0), device=kc.device)
    bd = TensorDesc(dtype=torch.int32, shape=(1, 4), stride=(4, 1), stride_order=(1, 0), device=dev)
    paged = dict(kv_source="paged", paged_page_size=16, k_cache_desc=kd, v_cache_desc=kd, block_table_desc=bd)
    with pytest.raises(NotImplementedError, match="prefill tile.*thd=True"):
        _Sdpa(geom, batch=1, seq_len=4, dtype=torch.bfloat16, device=dev, want_lse=False, **paged)
    st = _Sdpa(
        geom,
        batch=1,
        seq_len=4,
        dtype=torch.bfloat16,
        device=dev,
        want_lse=False,
        thd=True,
        num_sequences=1,
        max_seq_len=4,
        decode_route=decode_route(geom, 4),
        **paged,
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
            decode_route=decode_route(geom, 2),
            **paged,
        )


def test_the_envelope_names_the_policy_and_the_claimed_bound(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom, _ = _geom(16)
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
    d = env.decode
    assert d.served and d.policy == DEFAULT_DECODE_ROUTE_POLICY and d.claimed_max_s_q == DECODE_CLAIMED_MAX_S_Q and d.max_s_q == DECODE_MAX_S_Q
    assert d.route(2).claimed and d.route(2).decode_tile and d.route(4).prefill_tile and not d.route(4).claimed and d.route(8).prefill_tile
    assert not d.route(9).served and "decode_s_q" in d.declines and "decode_rows" in d.declines
    env_t = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy=DECODE_ROUTE_TOKEN_UNITS)
    assert env_t.decode.policy == DECODE_ROUTE_TOKEN_UNITS and env_t.decode.route(4).token_units == 2 and env_t.decode.route(4).claimed
    assert "S_q <= 8" in " ".join(d.requires) and "S_q <= 4" in " ".join(d.requires)
    with pytest.raises(ValueError, match="decode_route_policy must be one of"):
        GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107, decode_route_policy="fastest")


# ---------------------------------------------------------------------------
# Rubin: the cells
# ---------------------------------------------------------------------------

_LENS = (100, 384, 500, 0)  # one KV tile with a tail, the ring depth, four tiles with a tail, a PADDED entry (kv_lens > S_q on every live row)
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


# (group, S_q): the prefill route at the three MTP geometries -- 32 / 2 at S_q 4 / 5 / 8 (64 / 80 / 128 packed rows), 8 / 1 at S_q 5 / 8 (40 / 64 rows)
# -- and 24 / 2 at S_q 8 (the unpacked arm: the prefill tile cannot pack a 12-head group) and S_q 4 (two units: the policies coincide).
_ORACLE_CELLS = [(16, 4), (16, 5), (16, 8), (8, 5), (8, 8), (12, 4), (12, 8)]


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
    assert (r.tile, r.n_q, r.token_units, r.pack_gqa, r.kv_streams) == want[:5], (r, want)


@requires_rubin
def test_the_prefill_route_and_the_two_unit_route_agree_on_identical_inputs():
    """The A/B's two arms at 32 / 2, ``S_q = 4`` (the row the perf plan times) on IDENTICAL inputs (one seed: the same weights, inputs,
    prefix rows, pages and slots): the prefill route (one KV stream, the gate in the body) and the two-unit decode tile (two streams,
    the gate in the split combine) both within the oracle's budget and within one another's -- two kernels, two summation orders,
    so not bitwise; cos > 0.9999 between them and the max |diff| printed."""
    a = _case(("ab", 16, 4, "prefill"), 16, 4, _LENS, seed=31)
    b = _case(("ab", 16, 4, "two_unit"), 16, 4, _LENS, seed=31, blk_kw=dict(decode_route_policy=DECODE_ROUTE_TOKEN_UNITS))
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
    print(f"[32/2 S_q 4] prefill route vs two-unit route: cos {c:.6f} max|diff| {md:.3e}")
    assert c > 0.9999 and torch.equal(out_a[~live], out_b[~live]) and (out_a[~live] == 0).all()


@requires_rubin
def test_the_prefill_route_gate_fused_vs_the_separate_gate_launch():
    """``fuse_gate`` on the prefill route rides the paged body's epilogue (one launch fewer); against the same block with the gate as its own
    launch: both within the oracle's budget, ``out`` cos > 0.9999 between them."""
    fused = _case(("gate", 16, 5, "fused"), 16, 5, _LENS, seed=37)
    plain = _case(("gate", 16, 5, "launch"), 16, 5, _LENS, seed=37, blk_kw=dict(fuse_gate=False))
    assert fused.blk.gate_fused and fused.blk._gate is None and not plain.blk.gate_fused and plain.blk._gate is not None
    a, b = fused.run().clone(), plain.run().clone()
    fused.check_against_oracle(a, what="gate in the body")
    plain.check_against_oracle(b, what="gate launch")
    assert _readback(fused)["epilogue_gate"] == 1 and _readback(plain)["epilogue_gate"] == 0
    assert _cos(a[fused.live], b[fused.live]) > 0.9999


@requires_rubin
def test_two_launches_are_bitwise_and_the_workspace_is_honest_on_the_prefill_route():
    case = _case(("std", 16, 8), 16, 8, _LENS)
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
    lens = (300, 0, 1000, 77)
    case = _Case(16, 8, lens, seed=41)  # fresh: never executed
    assert case.blk.decode_route.prefill_tile
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
        sib = _Case(16, 8, (301, 0, 1000, 77), seed=43, table_row_stride=case.block_table.shape[1])
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
