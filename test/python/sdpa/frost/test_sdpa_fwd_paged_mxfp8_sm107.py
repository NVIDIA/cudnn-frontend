# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged MXFP8 pools on the cc 10.7 MXFP8 row (``sdpa_fwd_prefill_sm107_mxfp8``): the SM100 pool contract (#1214 --
F8_128x4 descale pools paging with K/V, page_size % 128, HND / NHD, V's own table, sinks and masks composed) served by
the ``PAGED_KV`` loader of ``sm107/prefill_d128_mxfp8.py`` / ``sm107/prefill_d256_mxfp8.py`` with dense queries.

What ``test_mhas_v2.py``'s cc 10.7 paged MXFP8 sweeps and the native-binder replay test do not cover: the Rule 3
sync-debug gate around ``execute`` on this row over pools, the sched-policy bit-identity under paging (d128: NATURAL /
LPT / LPT_L2 reorder whole work items, each tile's page walk is unchanged), e5m2-in / e4m3-out with Stats, pages of three
and four tiles, and the plan-time declines on the real device.  Builders, reference and O check are the SM100 suite's
(``test_sdpa_fwd_paged_sm100``, imported as helpers; its own cells stay pre-Rubin-gated).  The cc 10.7 CI lane runs an
explicit file list that does not include this module (its ``test_mhas_v2.py`` twins -- the plan-pin cells per scheduler
policy and the e4m3-O pins -- do run there); adding it to that list is a maintainers' follow-up."""

import pytest
import torch

from frost_test_utils import offers_engine, requires_dsl, requires_rubin, select_engine
from test_sdpa_fwd_paged_sm100 import _build_mxfp8, _check_o, _ref_mxfp8

pytestmark = [requires_rubin, requires_dsl, pytest.mark.L0]


def _run(B, H, KH, P, max_pages, lens, hnd, *, sched=None, **build_kw):
    """Build, pin the cc 10.7 MXFP8 row (optionally one scheduler policy), execute under Rule 3 and check O / LSE / Amax_O
    against the pool reference; -> (plan, O, LSE or None).  Empty sequences write O := 0 and LSE := sink (-inf without one)."""
    import cudnn
    import cudnn.sdpa  # noqa: F401 -- registers the FROST engines
    from cudnn.sdpa.fwd.engines import engine_name

    g, vp, Ob, stats_gpu, amax, ref_in = _build_mxfp8(B, H, KH, P, max_pages, lens, hnd, **build_kw)
    name = engine_name(arch="sm107", mxfp8=True)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    plan = select_engine(g, name)
    if sched is not None:  # pin one scheduler policy of the row's plans explicitly
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        idx = next((i for i, n in enumerate(names) if n.startswith(name) and g.plans[i].knobs.sched_policy == sched), None)
        assert idx is not None, f"no {name} plan with sched_policy={sched}; knobs={[p.knobs for p in g.plans]}"
        g.select_plan(idx)
        plan = g.plans[idx]
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")  # Rule 3: any blocking D2H in the FROST execute path is a bug
    try:
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    ref_o, ref_lse = _ref_mxfp8(**ref_in)
    out = Ob.float()
    assert not torch.isnan(out).any(), "NaN in O"
    atol = _check_o(out, ref_o, build_kw.get("out_dt", torch.bfloat16), build_kw.get("in_key", "e4m3"))
    assert abs(amax.item() - ref_o.abs().max().item()) <= atol, f"amax {amax.item()} vs ref {ref_o.abs().max().item()}"
    live = ref_in["seq_lens"] > 0
    if (~live).any():
        assert out[~live].abs().max().item() == 0.0, "empty sequence must write O := 0"
        if ref_in["sinks"] is not None:  # a keyless row with a sink: the sink is the whole softmax, LSE := sink
            ref_lse[~live] = ref_in["sinks"].view(1, H, 1).expand(B, H, ref_lse.shape[-1])[~live]
    lse = None
    if stats_gpu is not None:
        lse = stats_gpu.view(B, H, -1)
        torch.testing.assert_close(lse[live], ref_lse[live], atol=atol, rtol=3e-2)
        if (~live).any():
            torch.testing.assert_close(lse[~live], ref_lse[~live], atol=atol, rtol=3e-2)
    return plan, out.clone(), None if lse is None else lse.clone()


@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
def test_decode_pools(d, hnd):
    """Decode over pools with V behind its own table, dead pages NaN-poisoned, lengths incl. 0 and 2; a sink on the HND member."""
    _run(4, 8, 2, 128, 6, [700, 2, 129, 0], hnd, d_qk=d, d_v=d, separate_v=True, sink=hnd, stats=True)


def test_sink_swa_keyless_rows_d128():
    """Bottom-right causal + left window + sink at s_q 4 with a single-key and an empty sequence: keyless rows take O := 0 / LSE := sink."""
    _run(3, 8, 2, 128, 4, [1, 300, 0], False, d_qk=128, d_v=128, s_q=4, causal_br=True, window_left=128, sink=True, stats=True)


def test_e5m2_in_e4m3_out_stats_d256():
    _run(3, 8, 2, 128, 9, [1000, 1, 513], False, in_key="e5m2", out_dt=torch.float8_e4m3fn, stats=True)


def test_prefill_s512_stats_page256_d256():
    """Chunked prefill (s_q 512, bottom-right causal) over 256-row pages: two 128-row tiles per page."""
    _run(2, 8, 2, 256, 8, [1500, 700], True, s_q=512, causal_br=True, stats=True)


@pytest.mark.parametrize("P,d,s_q", [(384, 128, 8), (512, 256, 64)], ids=["p384-d128-mtp", "p512-d256-prefill"])
def test_three_and_four_tiles_per_page(P, d, s_q):
    """Pages of three (384) and four (512) 128-row tiles -- the TILES_PER_PAGE values the sweeps draw rarely (384) or never
    (512), so the tile_in_page / slot arithmetic and the SF descriptors' num_tiles extent are exercised beyond 1 / 2: MTP
    (d128, cga2) and chunked prefill (d256, cga1), bottom-right causal, sink, Stats, V behind its own table, a length one
    past a tile boundary (513) and one inside the third page (1000)."""
    _run(2, 8, 2, P, 3, [1000, 513], False, d_qk=d, d_v=d, s_q=s_q, causal_br=True, sink=True, stats=True, separate_v=True)


def test_sched_policies_bit_identical_under_paging_d128():
    """NATURAL / LPT / LPT_L2 reorder whole (batch, head, q-tile) work items; a tile's page walk is the same under each,
    so O and LSE are BIT-identical across the three policies over pools (the dense pin's paged twin)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

    runs = {
        p: _run(2, 8, 2, 128, 8, [1000, 513], False, d_qk=128, d_v=128, s_q=512, causal_br=True, stats=True, sched=p)
        for p in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2)
    }
    for p in (SCHED_LPT, SCHED_LPT_L2):
        assert runs[p][0].knobs.sched_policy == p
        assert torch.equal(runs[p][1], runs[SCHED_NATURAL][1]), f"O differs between NATURAL and policy {p}"
        assert torch.equal(runs[p][2], runs[SCHED_NATURAL][2]), f"LSE differs between NATURAL and policy {p}"


def test_declines_off_contract():
    """Plan-time declines stay plan-time on the real device: page 64, the d192x128 and d512 pools offer no cc 10.7 MXFP8 plan."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.sdpa.fwd.engines import engine_name

    def _offers(P, **kw):
        g, *_ = _build_mxfp8(2, 8, 2, P, 4, [100, 200], False, **kw)
        try:
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A])
        except (cudnn.cudnnGraphNotSupportedError, ValueError):
            return False
        return offers_engine(g, engine_name(arch="sm107", mxfp8=True))

    assert _offers(128, d_qk=128, d_v=128) and _offers(256, d_qk=256, d_v=256)
    assert not _offers(64, d_qk=128, d_v=128), "F8_128x4 pools hold whole 128-row SF atoms: page_size 64 is off-contract"
    assert not _offers(128, d_qk=192, d_v=128), "the d192x128 pools are a follow-up on cc 10.7"
    assert not _offers(128, d_qk=512, d_v=512), "the d512 pools are a follow-up on cc 10.7"


@pytest.mark.parametrize("sink", [False, True], ids=["nosink", "sink"])
@pytest.mark.parametrize("P,d", [(128, 256), (256, 256), (128, 128)], ids=["p128-d256", "p256-d256", "p128-d128"])
def test_masked_leading_tile_with_live_keys_behind_it(P, d, sink):
    """Q 65 x valid KV 193, bottom-right causal with left bound 34, attn_scale 1 (so |attn_scale * log2 e| >= 1), E5M2 in /
    f16 out, NHD pools: rows 33..64 see a fully masked FIRST KV tile and their legal keys only in the second one.  The
    scaled mask sentinel of that tile overflowed to -inf, the first tile's select seeded the running max with it and every
    later shift read -inf - (-inf) = NaN -- 65,536 nonfinite O elements at d256 (with a sink also 256 nonfinite Stats),
    while the scale-1/16 twin and the unwindowed sink control were finite (the #1481 review's native reproduction).  The
    four cc 10.7 MXFP8 bodies now clamp the scaled tile max to the finite sentinel, as the pre-folded arm always did.
    Dequantized Q / K / V = 0.5, so O is exactly 0.5 on every row with a legal key: only a NaN / inf (or a wrong LSE) can
    fail this cell, never FP8 rounding.  Dense twin: test_sdpa_fwd_mxfp8_sm100.py::test_masked_leading_tile_with_live_keys_behind_it."""
    from unittest.mock import patch

    with patch.object(torch, "randn", side_effect=lambda *a, **kw: torch.ones(*a, **kw)):
        _run(
            1,
            8,
            2,
            P,
            4,
            [193],
            False,
            s_q=65,
            causal_br=True,
            window_left=34,
            sink=sink,
            stats=True,
            in_key="e5m2",
            out_dt=torch.float16,
            d_qk=d,
            d_v=d,
            attn_scale=1.0,
        )
