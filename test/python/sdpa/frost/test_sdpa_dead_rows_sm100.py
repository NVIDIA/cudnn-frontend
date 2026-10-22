# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The STORED dead rows of the SM100 (cc 10.0 / 10.3) f16/bf16 small head-dim tiles under the dead-sub-tile QUIET body
(``sm100/prefill_d128_f16.py``, ``TemplateParams.dead_subtile_quiet = 1`` on the -inf body): a softmax sub-tile whose first row is at or
past its batch entry's Q length runs the quiet body -- the live body's barrier sequence and no math -- so its BMM2 multiplies whatever
the P region of the S slot holds (the S accumulator's fp32 bits read as bf16 pairs: NaN / Inf in roughly one low half per 256).  Without
per-batch Q lengths every dead row is at or past S_q and the O descriptor clips it; WITH them (``SEQ_Q_LENS_PRESENT``, the padded-Q trim),
and under bottom-right causal -- the only form under which the per-batch length reaches the softmax's dead-sub-tile decision
(``_resolve_seqlen_q`` folds to S_q otherwise) -- the rows in ``[seq_len_q[b], S_q)`` of a dead sub-tile ARE stored, and the epilogue's
trim must write them as EXACT zeros through a SELECT after the scale (``NaN * 0 = NaN``: a zero folded into ``inv_sum`` is not a zero
in O; the Stats are ``-inf`` by the same contract).

The route a caller takes there: a bottom-right causal graph with a kv_mask (a bias is not a graph under bottom-right), per-batch Q and KV
lengths, D 32 / 64, and the 64-wide KV tile through the public ``TILE_N`` autotune record (``sm100_dead_subtile_quiet_pick`` keeps the
full body on the 128-wide padded tile and admits the quiet body when ``S_q % (256 x cga)`` is in 1..128; a knob-less plan picks 128).

Tiers: the host contract (the pick on the detector's geometries, the adapter's ``template_params`` twin under a modelled cc 10.0 device,
the kernel source's select at the ``* inv_sum`` site); the GPU cells on a cc 10.0 / 10.3 device (the dev box is SHARED: functional only),
each a fresh graph on sentinel-filled outputs: the dead stored rows finite and exactly +0, every trimmed row 0 / -inf, the live rows
against the fp64 oracle, two launches bitwise -- RED on the tree before the select (NaN in O's dead rows) and GREEN with it; and the
Rubin (cc 10.7) twin's plain-body quiet form on its own line, whose epilogue already selects.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

import test_sdpa_bias_sm107 as _t7
from frost_test_utils import _SM, requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW_SM100 = "sdpa_fwd_prefill_sm100"
_ROW_SM107 = "sdpa_fwd_prefill_sm107"
requires_sm100_line = pytest.mark.skipif(_SM not in (100, 103), reason="needs a cc 10.0 / 10.3 device, have " + ("none" if _SM is None else f"sm_{_SM}"))

# The cells: bottom-right causal, the kv_mask word (30 % of the keys False), per-batch Q = KV lengths (square per batch), the 64-wide KV
# tile.  ``dead`` = (batch, lo, hi): the rows of the quiet sub-tile that are STORED (lo >= seq_len_q[b], hi <= S_q); ``request`` = the
# quiet body by request where the pick's rows rule returns 0 (S_q a multiple of 256 x cga).
_CASES = {
    # S_q 640 (640 % 256 = 128: the pick admits the quiet body): batch 1 ends at 300 -> item 1 (rows 256..511) keeps sub-tile 0 live (256 < 300)
    # and sub-tile 1 (384..511) dead AND stored; item 2 (512..767) collapses (the qtrim: its base row 512 >= 300).
    "mask_br_640_q300": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, bottom_right=True, q_lens=[640, 300], kv_lens=[640, 300], dead=(1, 384, 512)),
    # q_len_b = 384: sub-tile 0 of item 1 wholly live, sub-tile 1 wholly dead (no partially trimmed live sub-tile in between)
    "mask_br_640_q384": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, bottom_right=True, q_lens=[640, 384], kv_lens=[640, 384], dead=(1, 384, 512)),
    # the control: q_len_b = 256 -> item 1's base row is AT the length: the qtrim collapses the whole item (an EMPTY item: the zero-vector store, no
    # quiet iteration); nothing of the quiet body reaches a stored row
    "mask_br_640_q256_collapse": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, bottom_right=True, q_lens=[640, 256], kv_lens=[640, 256], dead=None),
    # S_q 512 (512 % 256 = 0: the rows rule returns 0 -- the quiet body by REQUEST, the adapter's dead_subtile_quiet=1): the audit's cell
    "mask_br_512_q300_request": dict(
        b=2, h=2, h_kv=2, s_q=512, s_kv=512, bottom_right=True, q_lens=[512, 300], kv_lens=[512, 300], dead=(1, 384, 512), request=1
    ),
    # the D 64 flavor at cga2: the cluster tile is 512 rows (640 % 512 = 128 admits the quiet body); the non-leader CTA's sub-tile 1 (384..511)
    # is the dead stored one; cluster tile 1 (512..1023) collapses
    "mask_br_640_q300_d64_cga2": dict(
        b=2, h=2, h_kv=2, s_q=640, s_kv=640, d=64, bottom_right=True, q_lens=[640, 300], kv_lens=[640, 300], dead=(1, 384, 512), cga=2
    ),
}
_TILE_N = 64


def _dead_rows(c):
    """(batch, lo, hi) of every row the trim stores as 0 / -inf: [seq_len_q[b], S_q) of each batch."""
    return [(b, q_len, c["s_q"]) for b, q_len in enumerate(c["q_lens"]) if q_len < c["s_q"]]


# --------------------------------------------------------------------------- tier 1: the host contract


def test_quiet_pick_admits_the_detector_geometries_and_keeps_the_full_body_off_them():
    """The pick on the cells' geometries: the quiet body on the 64-wide padded kv_mask body when S_q % (256 x cga) is in 1..128 (S_q 640 at
    cga1 and cga2), the full body on the knob-less 128-wide padded tile (the measured loss) and at S_q 512 (no dead sub-tile by the rows
    rule -- the per-batch length is not an input of the pick, so a request is the only route there)."""
    from cudnn.sdpa.fwd import config_sm100 as c0

    pick = c0.sm100_dead_subtile_quiet_pick
    assert pick(32, has_bias=False, has_kv_mask=True, s_q=640, padded=True, tile_n=64) == 1
    assert pick(64, has_bias=False, has_kv_mask=True, s_q=640, cga=2, padded=True, tile_n=64) == 1
    assert pick(32, has_bias=False, has_kv_mask=True, s_q=640, padded=True, tile_n=128) == 0, "the 128-wide padded body keeps the full body"
    assert pick(32, has_bias=False, has_kv_mask=True, s_q=512, padded=True, tile_n=64) == 0, "S_q % 256 == 0: the rows rule sees no dead sub-tile"
    assert pick(32, has_bias=False, has_kv_mask=True, s_q=512, padded=True, tile_n=64, request=1) == 1, "a request is honoured ahead of the rows rule"
    # the knob-less width of a padded kv_mask-only plan is 128: the exposure needs the TILE_N 64 autotune pick (a legal knob value)
    assert c0.sm100_tile_n_pick(32, 640, has_bias=False, has_kv_mask=True, padded=True, domain=frozenset({64, 128})) == 128


@pytest.fixture
def _sm100_device_for_the_adapter(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a cc 10.0 device on any host."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))


def _adapter(c, **kw):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    d = c.get("d", 32)
    q, o = (_t7._desc((c["b"], c["h"], c["s_q"], d), torch.bfloat16, n) for n in ("q", "o"))
    k, v = (_t7._desc((c["b"], c["h_kv"], c["s_kv"], d), torch.bfloat16, n) for n in ("k", "v"))
    mask = _t7._desc((c["b"], 1, 1, c["s_kv"]), torch.bool, "kv_mask")
    return SdpaFwdDslSm100(
        q,
        k,
        v,
        o,
        None,
        is_causal=True,
        causal_bottom_right=True,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        sample_kv_mask=mask,
        tile_n=_TILE_N,
        cga=c.get("cga"),
        dead_subtile_quiet=c.get("request"),
        **kw,
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_adapter_renders_the_quiet_body_with_the_trim_on_the_detector_geometries(_sm100_device_for_the_adapter, case):
    """The adapter's ``template_params`` (the row's twin, rule 8b) on every cell: the kv_mask arm, bottom-right, per-batch Q lengths
    (SEQ_Q_LENS_PRESENT), the 64-wide tile, the flavor's width -- and ``dead_subtile_quiet == 1`` (the pick, or the request where the
    rows rule returns 0): the GPU cells run the QUIET rendering of the trim arm, not the full body."""
    c = _CASES[case]
    api = _adapter(c)
    assert api.check_support() is True
    p = api.template_params()
    assert p.kv_mask_form != 0 and p.bias_mode == 0 and p.bottom_right and p.seq_q_lens_present and p.seq_kv_lens_present
    assert p.tile_n == _TILE_N and p.cta_mma == c.get("cga", 1) and p.d_flavor == c.get("d", 32)
    assert p.dead_subtile_quiet == 1, (case, p.dead_subtile_quiet)
    if c.get("request") is not None:
        assert _adapter(dict(c, request=None)).template_params().dead_subtile_quiet == 0, "without the request the rows rule keeps the full body at S_q 512"


def test_the_epilogue_selects_the_trimmed_rows_after_the_scale():
    """Static detector (sdpa-invariants s 2): the SM100 d128 file's O store multiplies ``o_chunk * inv_sum`` exactly once, and the
    per-batch-Q-length arm SELECTS ``row_trim | row_dead`` to zero right after it (``vec_select_zero``), before the cast -- never a
    zero folded into the scale alone."""
    from cudnn.sdpa.fwd.kernels.sm100 import prefill_d128_f16 as kmod
    from cudnn.frost.tile_dsl import pointwise

    src = inspect.getsource(kmod)
    sites = [i for i, ln in enumerate(src.splitlines()) if "o_scaled = o_chunk * inv_sum" in ln]
    assert len(sites) == 1, sites
    tail = "\n".join(src.splitlines()[sites[0] : sites[0] + 16])
    assert "if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):" in tail and "vec_select_zero(row_trim | row_dead, o_scaled, O_CHUNK)" in tail, tail
    assert tail.index("vec_select_zero(") < tail.index("o_fp16 = o_scaled.to(STORAGE_DTYPE)"), "the select precedes the cast"
    assert "arith.select(" in inspect.getsource(pointwise.vec_select_zero), "a select, not a multiply"


# --------------------------------------------------------------------------- tier 2: the GPU cells


def _run_twice(
    row, q, k, v, *, scale, bias=None, kv_mask=None, sdpa_kwargs=None, seq_len_kv=None, seq_len_q=None, sink=None, stats=True, cga=None, tile_n=None
):
    """Build the graph, pin the row's unsplit plan (a cga / tile_n request through the public autotune record), launch TWICE on
    SENTINEL-filled outputs (NaN: an unwritten row fails the finiteness check exactly like a residue does) and return both launches
    -- the two-launch identity is asserted by the caller AFTER the dead-row contract, so a NaN residue is reported as what it is."""
    import cudnn

    g, vp, o, o_gpu, stats_gpu = _t7._build(
        q, k, v, scale=scale, bias=bias, kv_mask=kv_mask, sdpa_kwargs=sdpa_kwargs, seq_len_kv=seq_len_kv, seq_len_q=seq_len_q, sink=sink, stats=stats
    )
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    ours = [i for i, n in enumerate(names) if n == row or n.startswith(row + "[")]
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {row} plan in {names}"
    picked = g.plans[idx].knobs
    picked_cga, picked_tile_n = picked.cga, picked.tile_n
    want_cga = picked_cga if cga is None else cga
    want_tile_n = picked_tile_n if tile_n is None else tile_n
    if (want_cga, want_tile_n) != (picked_cga, picked_tile_n):
        engine_id, record = g.get_engine_and_knobs_at_index(idx)
        record = dict(record)
        if want_cga != picked_cga:
            record[cudnn.knob_type.TILE_CGA_M] = want_cga
        if want_tile_n != picked_tile_n:
            record[cudnn.knob_type.TILE_N] = want_tile_n
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        o_gpu.fill_(float("nan"))
        if stats_gpu is not None:
            stats_gpu.fill_(float("nan"))
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return o_gpu.clone(), (stats_gpu.clone() if stats_gpu is not None else None)

    o1, st1 = launch()
    o2, st2 = launch()
    return dict(o=o1, stats=st1, o2=o2, stats2=st2, knobs=SimpleNamespace(cga=want_cga, tile_n=want_tile_n), names=names)


@pytest.fixture
def quiet_pick_spy(monkeypatch):
    """What the quiet pick returned for every plan the graph built, keyed by (tile_n, cga) -- ``api_dsl`` binds both arch lines' picks by
    name, so the spy sits there -- and, with ``request`` set, the adapter's request injected ahead of the rows rule (the standalone
    ``dead_subtile_quiet=`` kwarg the graph path has no knob for)."""
    from cudnn.sdpa.fwd import api_dsl

    spy = SimpleNamespace(calls=[], request=None)
    for name in ("_sm100_dead_subtile_quiet_pick", "_sm107_dead_subtile_quiet_pick"):
        real = getattr(api_dsl, name)

        def wrapped(*a, _real=real, **kw):
            if spy.request is not None:
                kw["request"] = spy.request
            r = _real(*a, **kw)
            spy.calls.append((int(kw.get("tile_n", 128)), int(kw.get("cga", 1)), int(r)))
            return r

        monkeypatch.setattr(api_dsl, name, wrapped)
    return spy


def _assert_dead_rows_contract(out, ref, c, case):
    """The contract, most specific first: the quiet sub-tile's stored rows finite and exactly +0 (the N1 rows), every trimmed row 0 / -inf,
    the whole-tensor contract against the oracle (keyless rows 0 / -inf, live rows within tolerance), two launches bitwise."""
    o, st = out["o"], out["stats"].squeeze(-1)
    if c["dead"] is not None:
        b, lo, hi = c["dead"]
        rows = o[b, :, lo:hi, :]
        n_nan = int((~torch.isfinite(rows)).sum())
        assert n_nan == 0, f"{case}: {n_nan} non-finite elements in the quiet sub-tile's stored rows O[{b}, :, {lo}:{hi}] (a residue reached the store)"
        assert bool((rows.view(torch.int16) == 0).all()), f"{case}: the quiet sub-tile's stored rows O[{b}, :, {lo}:{hi}] are not exactly +0"
        assert bool(torch.isneginf(st[b, :, lo:hi]).all()), f"{case}: Stats of the quiet sub-tile's stored rows are not -inf"
    for b, lo, hi in _dead_rows(c):
        assert bool(torch.isfinite(o[b, :, lo:hi, :]).all()) and bool(
            (o[b, :, lo:hi, :].view(torch.int16) == 0).all()
        ), f"{case}: trimmed rows O[{b}, :, {lo}:{hi}] != +0"
        assert bool(torch.isneginf(st[b, :, lo:hi]).all()), f"{case}: Stats of the trimmed rows [{lo}, {hi}) of batch {b} are not -inf"
    _t7._check(out, ref)
    assert torch.equal(out["o"], out["o2"]) and torch.equal(out["stats"], out["stats2"]), f"{case}: two-launch delta on O / Stats"


_GPU_CELLS = [(c, torch.bfloat16) for c in _CASES] + [("mask_br_640_q300", torch.float16)]


@requires_sm100_line
@pytest.mark.parametrize("case,dtype", _GPU_CELLS, ids=[f"{c}-{'bf16' if d == torch.bfloat16 else 'fp16'}" for c, d in _GPU_CELLS])
def test_trimmed_rows_of_a_quiet_dead_sub_tile_store_exact_zeros_on_sm100(case, dtype, quiet_pick_spy):
    """RED on the tree whose trim folded its zero into the scale (NaN x 0 = NaN in O[1, 384:512] of every cell but the collapse control),
    GREEN with the select after the scale.  The plan is the SM100 row's unsplit plan at the 64-wide tile (the autotune record), the quiet
    rendering (the spy reads the pick's 1 for that plan), on a fresh graph; the inputs are N(0, 1) everywhere (the rows past the length
    included: the S slot's bits are what the quiet sub-tile's BMM2 reads)."""
    c = _CASES[case]
    quiet_pick_spy.request = c.get("request")
    q, k, v, kw, _ = _t7._case_inputs(case, dtype, kv_mask_p=0.3, table=_CASES)
    kw["bias"] = None  # bottom-right causal + a bias is not a graph: the kv_mask arm alone carries the -inf body
    kw.pop("cga", None)
    ref = _t7._ref_of(c, q, k, v, kw)
    out = _run_twice(_ROW_SM100, q, k, v, tile_n=_TILE_N, cga=c.get("cga"), **kw)
    assert out["knobs"].tile_n == _TILE_N and out["knobs"].cga == c.get("cga", 1), (case, out["knobs"])
    assert (_TILE_N, c.get("cga", 1), 1) in quiet_pick_spy.calls, f"{case}: the plan did not take the quiet body: {quiet_pick_spy.calls}"
    _assert_dead_rows_contract(out, ref, c, case)


# The Rubin twin: the small tiles of cc 10.7 are the sm107 file, whose quiet body is the PLAIN body's (no bias, no kv_mask -- its pick is
# disjoint from the SM100 one) and whose epilogue selects ``_kv_empty | row_trim``; the same geometry without the mask, the heuristics' own
# width (the padded plain body's 64-wide tile) and pick (the default admits S_q 640).
_RUBIN_CASES = {
    "plain_br_640_q300": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, bottom_right=True, q_lens=[640, 300], kv_lens=[640, 300], dead=(1, 384, 512)),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_RUBIN_CASES))
def test_trimmed_rows_of_a_quiet_dead_sub_tile_store_exact_zeros_on_the_rubin_twin(case, quiet_pick_spy):
    c = _RUBIN_CASES[case]
    q, k, v, kw, _ = _t7._case_inputs(case, torch.bfloat16, table=_RUBIN_CASES)
    kw["bias"] = None
    kw.pop("cga", None)
    ref = _t7._ref_of(c, q, k, v, kw)
    out = _run_twice(_ROW_SM107, q, k, v, **kw)
    assert any(
        r == 1 and t == out["knobs"].tile_n for t, _g, r in quiet_pick_spy.calls
    ), f"{case}: the plan did not take the quiet body: {quiet_pick_spy.calls}"
    _assert_dead_rows_contract(out, ref, c, case)
