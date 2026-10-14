# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's DECODE (serving) form: a dense ``[B, S_q <= 4]`` declaration with ``kv_source="paged"`` over a
paged KV cache -- the step's tokens written through, then stage (4) over the first ``kv_lens[b]`` keys of every sequence's
``block_table`` row on the Rubin d256 decode tile (PackGQA over the whole group, the decode split model's split-KV with fp32
partials in the workspace, the sigmoid gate in the split COMBINE under ``fuse_gate``), the same tile UNPACKED (one Q head per
CTA) past the packed route up to 32 rows, and a typed decline beyond (no wider dense step has a kernel over the pools).

Host cells (any device, the declaration cells even on the CPU): the constants and the route table per ``(group, S_q)`` at the
four served groups (16 = 32 / 2, 12 = 24 / 2, 8 = 8 / 1, 6 = 24 / 4) against the module's route table; the static
``serving_envelope`` -- every decline the declaration raises is predicted by it, on this part and on a Rubin reading; the
declaration rows (typed, naming the feature); the declaration FACTS (route, split, where the gate runs, the stage list, the
capture-safe projections); the execute-time refusals on an UNCOMPILED block -- the Q-side length tensor ``seq_lens`` REFUSED
under ``kv_source="paged"``, ``block_table`` + ``kv_lens`` REQUIRED, form checks -- under ``set_sync_debug_mode("error")``;
the workspace carve unchanged by the read.

Rubin cells (one compiled block per cell, cached): every served group x ``S_q`` 1 .. 5 against the per-sequence fp32 oracle
(the block run on the FULL sequence, its last ``S_q`` rows; the prefix K / V of the pools ARE the oracle's bf16 rows), with
``kv_lens > S_q`` on every live row, a tail tile, one KV tile, ring depth 3 and 4, a padded entry (``kv_lens = 0`` + block 0 +
slot -1 -> ``out`` rows exactly 0, block 0 untouched -- the serving stacks' padded full-graph entry), ``S_q = 5`` on the
UNPACKED decode tile at groups 16 / 12 and on the packed two-token-unit tile at 8 / 6, the route pinned against the compiled
module; paged == dense on the same tokens (the standalone adapter over a padded dense cache gathered from the pools: BITWISE);
the block's form vs the standalone adapter on the SAME Q views / pools / table / gate: BITWISE; split == unsplit within the
combine's derived two-term budget (the pre-gate O of the block's own stage; LSE to fp32 rounding); the gate in the combine vs
the separate gate launch within the stated one-rounding budget (the exact max diff printed); L1 / L2 / L3 pools holding the
same values BITWISE on every page size (16 / 32 / 64 / 128) and f16; a block table that is a row SLICE of a larger table at an
arbitrary row stride (bitwise the compact table); CUDA-graph capture of the FIRST execute on a NEW stream with no warm-up and
the replay with new lengths / tables / slots / inputs bitwise the eager run; ``get_workspace_size()`` honest (the split
partials inside, a sentinel past the carve untouched); two launches bitwise.
"""

import contextlib
import math
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    DECODE_MAX_ROWS,
    DECODE_MAX_S_Q,
    KV_SOURCE_PAGED,
    KV_SOURCE_SELF,
    SERVING_PAGE_SIZES,
    SERVING_POOL_LAYOUTS,
    DecodeEnvelope,
    DecodeRoute,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    ServingEnvelope,
    decode_route,
)
from cudnn.gated_attention_block.api import DECODE_TILE, _cols, _Projection, _Sdpa, _SigmoidGate, _view, decode_split_choice

requirement_error = cutedsl_requirement_error("the gated attention block's serving tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, build_rope_tables, gated_attention_block_reference  # noqa: E402

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
# The route table the ADAPTER applies, (group, S_q) -> (N_Q, token units, packed): the PACKED route (the docs' decode route table
# -- the claimed envelope S_q <= 4 at every group; 8 / 1 rides the packed tile to S_q 8, 24 / 4 to S_q 10), then the
# adapter's UNPACKED arm (pack_g = 1: one head per CTA, one unit, S_q <= 16 on the 16-column tile, <= 32 on the 32-column one),
# then NOTHING: a dense step past 32 rows has no kernel over the pools (the adapter's decline -- an earlier reading, "the prefill kernel
# serves past the route", was wrong on both counts: the prefill body serves PACKED queries, the EXTEND form).
_ROUTE_TABLE = {
    16: {
        1: (16, 1, True),
        2: (32, 1, True),
        3: (32, 2, True),
        4: (32, 2, True),
        5: (16, 1, False),
        16: (16, 1, False),
        17: (32, 1, False),
        32: (32, 1, False),
        33: (0, 0, False),
    },
    12: {1: (16, 1, True), 2: (32, 1, True), 3: (32, 2, True), 4: (32, 2, True), 5: (16, 1, False), 33: (0, 0, False)},
    8: {1: (16, 1, True), 2: (16, 1, True), 3: (32, 1, True), 4: (32, 1, True), 5: (32, 2, True), 8: (32, 2, True), 9: (16, 1, False), 33: (0, 0, False)},
    6: {1: (16, 1, True), 2: (16, 1, True), 3: (32, 1, True), 5: (32, 1, True), 6: (32, 2, True), 10: (32, 2, True), 11: (16, 1, False), 33: (0, 0, False)},
    4: {
        1: (16, 1, True),
        4: (16, 1, True),
        5: (32, 1, True),
        8: (32, 1, True),
        9: (32, 2, True),
        16: (32, 2, True),
        17: (32, 1, False),
        32: (32, 1, False),
        33: (0, 0, False),
    },
}
_SENTINEL = 1024.0  # the pools' fill: FINITE (a serving stack's cache is zero-initialized; the tail tile reads and masks the rows past kv_len)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


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
    assert DECODE_MAX_S_Q == 4
    assert DECODE_TILE == "decode_d256_f16" and DECODE_MAX_ROWS == 32


@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_the_route_table_per_group_and_s_q_is_the_adapters_rule(group):
    """The decode route at every served group against the adapter's rule: the 16-column tile for <= 16 packed rows, the
    32-column tile in ``ceil(S_q / (32 // G))`` token units up to two, then the UNPACKED arm (one head per CTA) up to 32
    rows, NOT SERVED past that (the reason names the EXTEND form); the CLAIMED envelope is ``S_q <= 4`` everywhere, the
    wider cells routed but not claimed.  MHA stays unpacked."""
    geom, _ = _geom(group)
    for s_q, (n_q, units, packed) in _ROUTE_TABLE[group].items():
        r = decode_route(geom, s_q)
        assert isinstance(r, DecodeRoute) and r.s_q == s_q and r.group == group
        assert (r.n_q, r.token_units, r.pack_gqa) == (n_q, units, packed), (group, s_q, r)
        assert r.tile == (DECODE_TILE if n_q else None) and r.decode_tile is bool(n_q) and r.served is bool(n_q)
        assert (r.reason == "") is bool(n_q) and ("EXTEND form" in r.reason or n_q)
        assert r.claimed is (packed and s_q <= DECODE_MAX_S_Q)
        if packed:
            assert units == -(-s_q // (n_q // group))
        elif n_q:
            assert units == 1 and s_q <= n_q
    mha = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=4, h_kv=4, d_head=_D, rope_dim=_ROPE, causal_bottom_right=True)
    r = decode_route(mha, 2)
    assert (r.group, r.n_q, r.token_units, r.pack_gqa) == (1, 16, 1, False)
    d128 = GatedAttentionBlockGeometry(d_model=_D_MODEL, h_q=8, h_kv=2, d_head=128, rope_dim=_ROPE, causal_bottom_right=True)
    assert not decode_route(d128, 1).served and "d_head=128" in decode_route(d128, 1).reason  # no decode tile outside the d256 flavor


def test_the_split_choice_is_the_decode_split_model_floored_at_two_under_the_gate():
    from cudnn.sdpa.fwd.heuristics import choose_decode_tile_split_kv

    geom, _ = _geom(16)
    r = decode_route(geom, 2)
    for sm in (108, 148, 204, 212):
        want = choose_decode_tile_split_kv(units=4 * 2 * r.token_units, kv_tiles=8, sm_count=sm, q_tile=r.n_q)
        assert decode_split_choice(r, batch=4, h_kv=2, max_kv_len=1000, sm_count=sm, gated=False, h_q=32) == want
        assert decode_split_choice(r, batch=4, h_kv=2, max_kv_len=1000, sm_count=sm, gated=True, h_q=32) == max(want, 2)
    u = decode_route(geom, 5)  # the unpacked arm: one Q head per CTA
    assert not u.pack_gqa and u.decode_tile
    want_u = choose_decode_tile_split_kv(units=4 * 32, kv_tiles=8, sm_count=204, q_tile=16)
    assert decode_split_choice(u, batch=4, h_kv=2, max_kv_len=1000, sm_count=204, gated=False, h_q=32) == want_u
    assert decode_split_choice(decode_route(geom, 33), batch=4, h_kv=2, max_kv_len=1000, sm_count=204, gated=True, h_q=32) == 1  # an unserved route
    assert decode_split_choice(r, batch=4, h_kv=2, max_kv_len=1000, sm_count=0, gated=True) == 1  # no device: declined later anyway


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
    assert [r.s_q for r in d.routes] == list(range(1, 2 * DECODE_MAX_S_Q + 1))
    assert [r.tile for r in d.routes] == [DECODE_TILE] * 8 and [r.pack_gqa for r in d.routes] == [True] * 4 + [False] * 4
    assert d.route(3).token_units == 2 and not d.route(5).pack_gqa and d.route(32).served and not d.route(33).served and "EXTEND form" in d.route(33).reason
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
    assert env.forms == (("extend", "decode") if has_target else ())
    assert env.decode.route(1).served  # Geometry remains inspectable without a runnable target.
    if has_target:
        assert env.cutedsl_reason is None and env.extend.reason is None and env.decode.reason is None
    else:
        assert version in env.cutedsl_reason and "sm_107a" in env.cutedsl_reason
        assert env.extend.reason == env.decode.reason == env.cutedsl_reason


@pytest.mark.parametrize(
    ("cc", "match"),
    [((8, 0), "found SM80"), ((10, 0), "found SM100"), ((10, 3), "found SM103"), (None, "no CUDA device")],
    ids=["sm80", "sm100", "sm103", "none"],
)
def test_the_envelope_declines_every_other_part_without_raising(cc, match):
    geom, _ = _geom(16)
    env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, "cpu" if cc is None else _DEV, device_cc=cc)
    assert not env.decode.served and not env.extend.served and env.forms == () and match in env.decode.reason and match in env.extend.reason
    assert env.decode.routes  # the routes are geometry facts, listed either way


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
    assert GatedAttentionBlockFwd.serving_envelope(geom16, torch.bfloat16, _DEV).decode.served is (_cc() == _SM107)


# ---------------------------------------------------------------------------
# The declaration: rows, facts
# ---------------------------------------------------------------------------


def test_declaration_rows_are_typed_and_name_the_feature():
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
    # past the unpacked tile's 32 rows a dense step has no kernel over the pools: the adapter's decline, raised at declaration
    with pytest.raises(NotImplementedError, match="decode tile only.*EXTEND form"):
        _make_block(16, 2, 33)
    blk32, _ = _make_block(16, 2, 32)
    assert blk32.decode_route.served and not blk32.decode_route.pack_gqa and blk32.decode_route.n_q == 32
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
def test_declaration_facts(group, s_q):
    """A declared DECODE block records its route, its split and where the gate runs BEFORE anything compiles; the stage list
    is the serving pipeline; both projections are capture-safe; ``seq_lens_present`` reads True whatever was passed."""
    # Pin the split off CUDA: sm_count=0 leaves the gate outside the combine.
    blk, _ = _make_block(group, 4, s_q, fuse_gate=True, fuse_norm_rope=True, decode_split_kv=None if torch.cuda.is_available() else 2)
    r = blk.decode_route
    assert r == decode_route(blk.geom, s_q) and blk.kv_source == "paged" and blk.max_kv_len == 64 * 16 and blk.seq_lens_present is True
    assert blk._sdpa.paged_dense and blk._sdpa.decode_route == r and blk._sdpa.decode_split_kv == blk.decode_split_kv and blk._sdpa.seq_lens_present
    assert r.decode_tile and blk.decode_split_kv >= 2 and blk.gate_fused and blk._gate is None and blk._sdpa.fuse_gate
    names = [st.name for st in blk._stages]
    # the write-through is FOLDED into the fused projection's epilogue under AUTO (fuse_cache_write=None): no cache_write stage
    assert blk.cache_write_fused and blk._cache_write is None and blk._proj.kv_pool_page_size == blk.paged_kv_page_size
    assert names == ["qkv_gate_proj_norm_rope", "sdpa"] + ([] if blk.gate_fused else ["sigmoid_gate"]) + ["out_proj"], names
    blk_s, _ = _make_block(group, 4, s_q, fuse_gate=True, fuse_norm_rope=True, decode_split_kv=None if torch.cuda.is_available() else 2, fuse_cache_write=False)
    assert not blk_s.cache_write_fused and [st.name for st in blk_s._stages] == ["qkv_gate_proj_norm_rope", "cache_write", "sdpa"] + (
        [] if blk_s.gate_fused else ["sigmoid_gate"]
    ) + ["out_proj"]
    assert blk._out_proj.capture_safe is True and (not isinstance(blk._proj, _Projection) or blk._proj.capture_safe is True)
    # the pinned split, the unfused gate, the unfused norm + RoPE
    blk1, _ = _make_block(group, 4, s_q, fuse_gate=True, decode_split_kv=1)
    assert blk1.decode_split_kv == 1 and not blk1.gate_fused and isinstance(blk1._gate, _SigmoidGate) and not blk1._sdpa.fuse_gate
    assert [st.name for st in blk1._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write", "sdpa", "sigmoid_gate", "out_proj"]
    blk4, _ = _make_block(group, 4, s_q, fuse_gate=False, decode_split_kv=4)
    assert blk4.decode_split_kv == 4 and blk4._sdpa.decode_split_kv == 4 and not blk4.gate_fused


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


def test_the_stage_refuses_the_paged_inputs_on_a_self_declaration_and_the_q_side_length_on_a_paged_one():
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
    with pytest.raises(NotImplementedError, match="decode tile only.*EXTEND form"):
        _Sdpa(geom, batch=1, seq_len=33, dtype=torch.bfloat16, device=dev, want_lse=False, **paged)
    with pytest.raises(NotImplementedError, match="split combine only"):
        _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=False, fuse_gate=True, decode_split_kv=1, **paged)
    with pytest.raises(NotImplementedError, match="writes no LSE"):
        _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=True, **paged)
    st = _Sdpa(geom, batch=1, seq_len=2, dtype=torch.bfloat16, device=dev, want_lse=False, decode_split_kv=2, fuse_gate=True, **paged)
    assert st.paged_dense and st.decode_route.decode_tile and st.seq_lens_present and st.decode_split_kv == 2


def test_the_workspace_carve_is_unchanged_by_the_paged_read():
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
    ):
        self.group, self.s_q, self.kv_lens, self.page, self.layout, self.dtype = group, int(s_q), [int(x) for x in kv_lens], page, layout, dtype
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
        self.h_full = [randn(1, max(L, s_q), self.d_model, std=1.0) for L in self.kv_lens]
        # the step's inputs: the LAST s_q rows (and their positions' RoPE rows); a padded entry takes its row 0 .. s_q-1 (never read)
        self.h = torch.stack([hf[0, -s_q:] for hf in self.h_full])  # [B, s_q, d_model]
        pos = [list(range(L - s_q, L)) if L else list(range(s_q)) for L in self.kv_lens]
        self.cos = torch.stack([cos_t[0, p] for p in pos])
        self.sin = torch.stack([sin_t[0, p] for p in pos])
        # the oracle per live entry over the FULL sequence: its K / V rows (bf16 -- the oracle's single rounding) fill the pools
        self.ref_out = torch.zeros(B, s_q, self.d_model, dtype=dtype, device=_DEV)
        self.ref_rows = {}
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
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
            fuse_norm_rope=dtype == torch.bfloat16,
        )
        kw.update(blk_kw or {})
        self.blk_kw = kw
        self.out = torch.empty(B, s_q, self.d_model, dtype=dtype, device=_DEV)
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, self.geom, **kw
        )
        self.ws = None

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

    def check_against_oracle(self, out=None, *, what="out"):
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
    """The DECODE cells of the served-group matrix: every served group x S_q 1 .. 4 (the claimed envelope) and S_q = 5 (past the packed route at 16 /
    12: the adapter's UNPACKED tile, one head per CTA; the packed tile's second token unit at 8 / 6 -- correct and DECLARED),
    against the per-sequence fp32 oracle under the dense
    suite's budget (cos > 0.999 on ``out``), the padded entry EXACTLY 0 with block 0 untouched, the route pinned against the
    compiled kernel (its template, packing, split and the gate OUT of the kernel: in the combine)."""
    case = _std_case(group, s_q)
    case.run()
    case.check_against_oracle()
    r, rb = case.blk.decode_route, case.route_readback()
    assert rb["template"] == r.tile, rb
    assert rb["adapter_split"] == case.blk.decode_split_kv and (rb["adapter_pack"] is True) == r.pack_gqa, (rb, r)
    assert r.decode_tile
    if True:
        assert rb["file"].endswith("/sm107/decode_d256_f16.py") and rb["heads_per_tile"] == (group if r.pack_gqa else 1), rb
        assert rb["n_q"] == r.n_q and rb["split"] == case.blk.decode_split_kv, rb
        assert rb["epilogue_gate"] is False and case.blk.gate_fused and case.blk.decode_split_kv >= 2, "the gate rides the COMBINE, never the tile"
    assert r.claimed is (s_q <= DECODE_MAX_S_Q and r.pack_gqa)
    k0, v0 = case.block0_bits
    assert torch.equal(_bits(case.k_cache[0]), k0) and torch.equal(_bits(case.v_cache[0]), v0), "block 0 (the NULL block) must never be written"


@requires_rubin
def test_s_q_five_rides_the_unpacked_tile_at_sixteen_and_twelve_and_the_packed_tile_at_eight_and_six():
    """Past the PACKED route (S_q = 5 at 32 / 2 and 24 / 2) the adapter's rule rides the decode tile UNPACKED -- one head per
    CTA (HEADS_PER_TILE 1), one token unit -- not the prefill body an earlier route table named; at 8 / 1 and 24 / 4 the packed
    tile's second token unit serves it, at 4 / 1 one unit of the 32-column tile (20 packed rows).  Every arm declared by the
    route and pinned on the compiled module."""
    for group in sorted(_GROUPS):
        case = _std_case(group, 5)
        r, rb = case.blk.decode_route, case.route_readback()
        assert r.tile == DECODE_TILE == rb["template"] and r.pack_gqa is (group in (8, 6, 4)), (group, r, rb)
        assert rb["heads_per_tile"] == (group if r.pack_gqa else 1) and rb["adapter_pack"] is (True if r.pack_gqa else False)


def test_past_thirty_two_rows_the_step_is_a_typed_decline_the_envelope_predicts():
    """A dense step wider than the 32-column tile has no kernel over the pools on this part (the adapter: "Rubin paged KV serves
    half THD queries, MXFP8 pools on D128 / D256, and decode-shaped half D256 graphs on the decode tile"): the route says not
    served with the reason (the EXTEND form), the declaration raises it, the envelope predicts it -- no silent fall-through to
    a prefill body an earlier route table assumed."""
    for group in sorted(_GROUPS):
        geom, _ = _geom(group)
        r = decode_route(geom, DECODE_MAX_ROWS + 1)
        assert not r.served and r.tile is None and "EXTEND form" in r.reason and r.n_q == 0
        assert decode_route(geom, DECODE_MAX_ROWS).served
        with pytest.raises(NotImplementedError, match="decode tile only.*EXTEND form"):
            _make_block(group, 2, DECODE_MAX_ROWS + 1)
        env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
        assert not env.decode.route(DECODE_MAX_ROWS + 1).served and env.decode.max_rows == DECODE_MAX_ROWS


# The 397B shards at their REAL width (d_model 4096; the layout contract pins n_qkvg 8704 / 4608 / 2560): 16 / 1 (TP 2) rides group
# 16's route with ONE KV head, 8 / 1 (TP 4, the serving deployment) group 8's, 4 / 1 (TP 8) group 4's -- one kernel block size per
# shard, so the three cells also walk the page sizes the deployments allocate.  (h_q, h_kv, page)
_SHARDS = {"q35_397b_tp2_16_1": (16, 1, 16), "q35_397b_tp4_8_1": (8, 1, 32), "q35_397b_tp8_4_1": (4, 1, 64)}
_D_MODEL_397B = 4096


def test_the_397b_shards_ride_their_groups_routes_and_the_envelope_serves_both_forms(monkeypatch):
    """A shard's decode route is its GROUP's (the rule reads ``h_q // h_kv`` and ``d_head`` only, never ``d_model`` or the KV head
    count): 16 / 1 = the 32 / 2 rows, 8 / 1 = group 8's, 4 / 1 = group 4's (packed to ``S_q = 16`` in two token units, unpacked to
    32, a typed decline past that); the envelope on a Rubin reading serves the EXTEND and the DECODE form at every shard."""
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
        for s_q, (n_q, units, packed) in _ROUTE_TABLE[group].items():
            r = decode_route(geom, s_q)
            assert (r.n_q, r.token_units, r.pack_gqa) == (n_q, units, packed), (name, s_q, r)
        env = GatedAttentionBlockFwd.serving_envelope(geom, torch.bfloat16, _DEV, device_cc=_SM107)
        assert env.forms == ("extend", "decode") and env.extend.served and env.decode.served, (name, env.extend.reason, env.decode.reason)
        assert env.decode.route(1).claimed and env.decode.route(DECODE_MAX_S_Q).claimed and not env.decode.route(DECODE_MAX_ROWS + 1).served


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
    pools, table, gate, split and packing: BITWISE (the block hands the adapter exactly these views)."""
    case = _std_case(16, 3)
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


@requires_rubin
def test_capture_of_the_first_execute_replays_bitwise_and_takes_new_lengths_tables_slots_and_inputs():
    """The serving stack's defaults: the FIRST execute of a fresh declaration captured on a NEW stream with no eager warm-up
    under ``set_sync_debug_mode("error")`` (no host sync in the DECODE path: lengths, tables and slots are device data); the
    capture launches nothing, the replay is BITWISE a fresh eager run; a padded batch replays correctly; new ``kv_lens`` /
    ``block_table`` / ``slot_mapping`` / ``h`` through the captured tensors replay bitwise the eager run over them."""
    lens = (300, 0, 1000, 77)
    case = _Case(16, 2, lens, seed=23)  # fresh: never executed
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
        sib = _Case(16, 2, (301, 0, 1000, 77), seed=29, table_row_stride=case.block_table.shape[1])
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
