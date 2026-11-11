# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM90 native D128 forward template (issue #378), through the direct adapter and the graph.

``SdpaFwdDslSm90`` runs ``sm90/prefill_d128_f16.py`` for an exact (128, 128) dense declaration
it serves (``config_sm90.pick_flavor``). The direct-route cases build the adapter and assert the
D128 flavor before they trust a number; O and Stats start as NaN, so a row the kernel never writes
cannot pass. The graph-route cases (the engine row proposes the 128 x 128 tile, the D128 template,
first on the exact (128, 128) graphs it serves) reuse the SM100 suite's runners through the SM90
suite's ``sm100`` fixture, which pins the SM90 row strictly, and check that every such graph
compiles the D128 template, never D512; the tile's declines at (128, 128) are checked on the
ranked graph, each beside a (512, 512) control. Tolerances are the
SM90 suite's own (O 5e-2 / 3e-2, Stats 2e-2 / 2e-2 against the fp32 reference; the reused SM100
runners keep theirs) except where the arithmetic is the same and the comparison is ``torch.equal``.
"""

import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch

from test_utils import torch_fork_set_rng

import cudnn
from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from frost_test_utils import _SM, assert_no_new_spills, nvdisasm_candidates, offers_engine, requires_dsl, select_engine
from test_sdpa_fwd_dsl_sm90 import sm100  # noqa: F401 (the strict SM90 graph-route fixture)
from test_sdpa_fwd_dsl_sm100 import _bhsd, _ref_sdpa_full

pytestmark = [
    pytest.mark.skipif(_SM != 90, reason="needs an SM90 Hopper GPU, have " + ("none" if _SM is None else f"sm_{_SM}")),
    requires_dsl,
]


# Levels (test/AGENTS.md: L0 stays fast, big sweeps go to higher levels). Every test carries its own level mark, since a
# module-level L0 would select every parameter of a sweep. The sweeps below keep one or a few cells per record type in L0
# (the one-tile kernel, the persistent loop, the paired walk, the NATURAL stride, the claim counter) and run the rest of
# their forced-grid cells, the other SASS probes and the other watchdogged children at L1.
_L0_CROSS = frozenset({"ring-wrap", "causal-br", "gqa-causal"})
_L0_WALK = frozenset({"pairs-t10", "middle-light-lone"})
_L0_STRIDE = frozenset({("t1-tl-mha", 2), ("t1-tl-mha", 3), ("t8-br-gqa", 2), ("t8-br-gqa", 5)})
_L0_CLAIM = frozenset({"t8-br-gqa", "t32-tl-gqa"})
_L0_SASS = frozenset({"pc1-causal"})  # carries the claim ring's release-dependence pin
_L0_WATCHDOG = frozenset({"pa"})


def _level(l0):
    return pytest.mark.L0 if l0 else pytest.mark.L1


_D = 128
_SCALE = 1.0 / math.sqrt(_D)
_ROW = "sdpa_fwd_prefill_sm90"
_D128_FILE = "sm90/prefill_d128_f16.py"
_DTYPES = (torch.float16, torch.bfloat16)


def _inputs(b, h_q, h_kv, s_q, s_kv, dtype):
    """Q, K, V as BHSD views over BSHD storage (the notebook convention)."""
    return _bhsd(b, h_q, s_q, _D, dtype), _bhsd(b, h_kv, s_kv, _D, dtype), _bhsd(b, h_kv, s_kv, _D, dtype)


# The overlap, persistent, causal walk and claim counter switch fields (config_sm90.TemplateParams). A variant names the
# ones it turns on; _plan sets every other one off (False, or 0 for the causal walk and the claim counter), so a variant
# never inherits the adapter's shipped values.
_SWITCHES = ("intra_wg_overlap", "persistent", "causal_walk", "causal_claim")
_VARIANTS = {"v1": (), "ov": ("intra_wg_overlap",), "pa": ("intra_wg_overlap", "persistent")}
# The persistent variants: each must compute ov's O and Stats bit for bit.
_PERSISTENT = ("pa",)
# Variants of causal records only: the paired walk (causal_walk 1), which validate_params refuses on a dense record.
_CAUSAL_VARIANTS = {"pw": ("intra_wg_overlap", "persistent", "causal_walk")}
# The claim counter's variant, causal NATURAL records only (causal_claim 1: a group's Q tiles in NATURAL order):
# persistent under overlap, the walk off. A claim record needs the caller workspace (_claim_run).
_CLAIM_VARIANTS = {"pc1": 1}
# Forced persistent CTA counts: one CTA crossing every item boundary, two and three wrapping the K/V rings and the Q phase
# at uneven parities, 131 leaving one SM's worth of items over. A cross-item bug shows only when a CTA runs more than one
# item (python/cudnn/sdpa/AGENTS.md, the THD re-seed lesson): at the adapter's count every small case runs one item per CTA.
_FORCED_GRIDS = (1, 2, 3, 131)


def _switches(variant):
    """``variant``'s whole switch record: each _SWITCHES field on exactly when the variant names it (True, or 1 for the
    causal walk's int); a claim variant is the persistent overlap record with its causal_claim value and the walk off."""
    if variant in _CLAIM_VARIANTS:
        return dict(intra_wg_overlap=True, persistent=True, causal_walk=0, causal_claim=_CLAIM_VARIANTS[variant])
    names = {**_VARIANTS, **_CAUSAL_VARIANTS}[variant]
    return {name: int(name in names) if name in ("causal_walk", "causal_claim") else name in names for name in _SWITCHES}


def _counts(variant):
    """The watchdog child's persistent CTA counts: the adapter's (0), and for a persistent variant 1 and 2 (one CTA crossing
    every item boundary; two wrapping the K/V rings and the Q phase at uneven parities)."""
    return [0, 1, 2] if variant in _CLAIM_VARIANTS or "persistent" in {**_VARIANTS, **_CAUSAL_VARIANTS}[variant] else [0]


def _plan(q, k, v, *, stats=False, variant=None, grid=None, **kw):
    """One compiled D128 plan over NaN-filled O (and Stats); asserts the adapter chose D128.

    ``variant`` names a switch record (``_VARIANTS``), set on the plan's record between
    ``check_support`` and ``compile``; the compiled module must carry every switch of it, so a cache-key
    slip that hands back another record's kernel fails here. None is the adapter's shipped record.
    ``grid`` forces a persistent record's CTA count (the instance attribute shadows
    ``SdpaFwdDslSm90._persistent_ctas``; each count is its own compile key).
    """
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h_q, s_q, d = q.shape
    o = torch.full((b, s_q, h_q, d), float("nan"), dtype=q.dtype, device="cuda").transpose(1, 2)
    lse = torch.full((b, h_q, s_q), float("nan"), dtype=torch.float32, device="cuda") if stats else None
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, **kw)
    assert api.check_support() and api.flavor == (128, 128), api.flavor
    if variant is not None:
        api.params = dataclasses.replace(api.params, **_switches(variant))
    if grid is not None:
        api._persistent_ctas = lambda device, n=grid: n
    api.compile()
    if variant is not None:
        compiled = {name: getattr(api._k_mod, name.upper()) for name in _SWITCHES}
        assert compiled == _switches(variant), (variant, compiled)
    return api, o, lse


def _run(q, k, v, *, stats=False, variant=None, grid=None, **kw):
    """Plan, execute once, and return (O, Stats or None)."""
    api, o, lse = _plan(q, k, v, stats=stats, variant=variant, grid=grid, **kw)
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    torch.cuda.synchronize()
    return o, lse


def _check(o, lse, q, k, v, *, scale=_SCALE, is_causal=False, bottom_right=False):
    o_ref, lse_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=is_causal, bottom_right=bottom_right, return_stats=True)
    assert not torch.isnan(o).any(), "an O element kept its NaN prefill: a row or column was never stored"
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    if lse is not None:
        assert not torch.isnan(lse).any(), "a Stats row kept its NaN prefill"
        torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=2e-2)


# --- Routing -------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_compiles_its_own_template(monkeypatch):
    """The direct adapter loads exactly the D128 file under its own tag, and the result is right."""
    from cudnn.sdpa.fwd import api_dsl

    loaded, load = [], api_dsl._load_kernel_template
    monkeypatch.setattr(
        api_dsl,
        "_load_kernel_template",
        lambda filename, params, tag="template", part=None: loaded.append((filename, tag)) or load(filename, params, tag, part=part),
    )
    q, k, v = _inputs(2, 4, 4, 96, 160, torch.float16)
    o, lse = _run(q, k, v)
    assert loaded == [(_D128_FILE, "sdpa_fwd_sm90_d128")], loaded
    _check(o, lse, q, k, v)


# --- The graph route: the engine row offers exact (128, 128) graphs -------
#
# The SM100 suite's runners, pinned to the SM90 row by the ``sm100`` fixture. ``graph_d128`` adds the silent-D512
# detector (#378): a (128, 128) graph whose first plan is the 64 x 64 tile runs on the D512 body without any error,
# so a graph the heuristics stopped proposing on the 128 x 128 tile would pass on D512. Each graph_d128 case must
# load a template, and only D128's.


@pytest.fixture
def graph_d128(sm100, monkeypatch):
    """``sm100``, plus a record of every template the case loads, checked when the case ends: D128's file only."""
    from cudnn.sdpa.fwd import api_dsl

    loaded, load = [], api_dsl._load_kernel_template
    monkeypatch.setattr(api_dsl, "_load_kernel_template", lambda filename, *a, **kw: loaded.append(filename) or load(filename, *a, **kw))
    yield sm100
    assert loaded and set(loaded) == {_D128_FILE}, f"a (128, 128) graph loaded {loaded}, not only {_D128_FILE}"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_graph_compiles_the_d128_template(graph_d128, monkeypatch):
    """A (128, 128) graph the row admits builds a plan whose executor names its template (``kernel_template``, which
    the SM90 adapter records as the SM100 one does) and keeps its module: SM100 ships a prefill_d128_f16.py too, so the
    sm90 directory is the evidence. The fixture checks every load."""
    graphs, pin = [], graph_d128._select_engine
    monkeypatch.setattr(graph_d128, "_select_engine", lambda graph, name, **kw: graphs.append(graph) or pin(graph, name, **kw))
    q, k, v = (_bhsd(2, 4, 96, _D, torch.float16) for _ in range(3))
    o = graph_d128._run_dsl_graph(q, k, v, scale=_SCALE, dtype=torch.float16, sdpa_kwargs=dict(use_causal_mask=True))
    torch.testing.assert_close(o, _ref_sdpa_full(q, k, v, scale=_SCALE, is_causal=True), atol=5e-2, rtol=3e-2)
    (graph,) = graphs
    executor = graph._compiled_plans[graph._plan_index]._compiled
    assert executor.kernel_template == "prefill_d128_f16", executor.kernel_template
    assert executor.kernel_module.__file__.endswith(os.path.join("sm90", "prefill_d128_f16.py")), executor.kernel_module.__file__


@pytest.mark.L0
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_d128_graph_api(graph_d128, dtype, causal):
    """The canonical smoke case, repinned: no length tensor and no Stats, so the LSE store compiles out and the runner
    asserts a zero workspace."""
    graph_d128.test_sdpa_fwd_dsl_sm100_graph_api(dtype, causal, _D)


_GRAPH_MASKS = ("none", "causal", "causal_br")  # the D128 subset of the SM100 suite's dense mask families


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(8, 8), (8, 2)], ids=["mha", "gqa4"])
@pytest.mark.parametrize("i, mask", list(enumerate(_GRAPH_MASKS)), ids=list(_GRAPH_MASKS))
@torch_fork_set_rng(seed=0)
def test_d128_graph_mask_families(graph_d128, i, mask, heads):
    """Each mask D128 serves, MHA and GQA; the dtype rotates with the mask (causal_br runs S_q 128 below S_kv 256)."""
    graph_d128._combo_dense(_D, _DTYPES[i % 2], *heads, _SCALE, None, mask)


@pytest.mark.L0
@pytest.mark.parametrize("stats_log2", [False, True], ids=["ln", "log2"])
@torch_fork_set_rng(seed=59)
def test_d128_graph_strided_stats(graph_d128, stats_log2):
    """Dense multi-row Stats in permuted, gapped storage: the store reads all three declared strides, so the strided
    Stats equal the contiguous ones bit for bit, in either base."""
    graph_d128._check_dsl_sm100_strided_stats(_D, _D, stats_use_log2=stats_log2)


@pytest.mark.L0
@pytest.mark.parametrize("scale", [-0.25, 0.0], ids=["negative", "zero"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "unmasked"])
@torch_fork_set_rng(seed=0)
def test_d128_graph_scale_signs(graph_d128, scale, causal):
    """The graph's attn_scale sign reaches the D128 specialization, masked and unmasked (the SM90 suite's
    test_dsl_sm90_scale_sign runs causal at D512 only)."""
    q, k, v = (_bhsd(2, 4, 100, _D, torch.float16) for _ in range(3))
    sdpa_kwargs = dict(use_causal_mask=True) if causal else {}
    o, lse = graph_d128._run_dsl_graph(q, k, v, scale=scale, dtype=torch.float16, sdpa_kwargs=sdpa_kwargs, return_stats=True)
    o_ref, lse_ref = _ref_sdpa_full(q, k, v, scale=scale, is_causal=causal, return_stats=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse.squeeze(-1), lse_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_graph_omitted_scale_is_no_scaling(graph_d128):
    """An omitted attn_scale is 1.0 on the graph, as the backend lowers it, never the adapter's 1/sqrt(d) default."""
    graph_d128.test_dsl_sm100_omitted_attn_scale_is_no_scaling(_D)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_graph_softmax_precision_float_is_the_default_arithmetic(graph_d128):
    """sdpa(softmax_precision=FLOAT) names the arithmetic D128 always runs, so O and Stats equal the unset graph's bit
    for bit. A set precision makes the node backend-unlowerable: only FROST can serve this graph."""
    q, k, v = (_bhsd(2, 4, 100, _D, torch.bfloat16) for _ in range(3))

    def run(**kw):
        sdpa_kwargs = dict(use_causal_mask=True, **kw)
        return graph_d128._run_dsl_graph(q, k, v, scale=_SCALE, dtype=torch.bfloat16, sdpa_kwargs=sdpa_kwargs, return_stats=True)

    o, lse = run()
    o_float, lse_float = run(softmax_precision=cudnn.data_type.FLOAT)
    assert torch.equal(o, o_float) and torch.equal(lse, lse_float)
    o_ref, lse_ref = _ref_sdpa_full(q, k, v, scale=_SCALE, is_causal=True, return_stats=True)
    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(lse.squeeze(-1), lse_ref, atol=2e-2, rtol=2e-2)


# --- The graph route: what the 128 x 128 tile declines at (128, 128) -------
#
# Each case is a whole SM100 runner whose strict pin is swapped for a probe: the graph is built and ranked as for a
# launch, then the probe reads whether the SM90 row offers it and the row's reason to decline the 128 x 128 tile (the
# D128 template), and stops before any build. Each graph keeps the 64 x 64 plans (the D512 body); the (512, 512)
# control shows the same graph is the row's there too.


class _Ranked(Exception):
    """Raised by the probe once the runner's graph is ranked: nothing is built or launched."""


def _ranking(sm100, monkeypatch, check):
    """Swap the runners' pin for ``check(graph)``, then stop; return a function that runs one runner and gives back the
    check's value. A graph no engine at all can plan raises before the pin, with every python engine's reason in its
    message, so that message stands for the row's reason."""
    seen = []

    def probe(graph, name, **_):
        assert name == _ROW, name
        seen.append(check(graph))
        raise _Ranked

    monkeypatch.setattr(sm100, "_select_engine", probe)

    def rank(run):
        try:
            run()
        except _Ranked:
            return seen.pop()
        except cudnn.cudnnGraphNotSupportedError as exc:
            return False, str(exc)
        raise AssertionError("the runner never reached its pin")

    return rank


def _row_verdict(graph):
    """(offered, the row's reason to decline the 128 x 128 tile or None) for a ranked graph."""
    from cudnn.sdpa.fwd import engines

    spec = next(s for s in engines.ENGINE_SPECS if s.name == _ROW)
    facts = engines.analyze_for(spec, graph)[0]
    return offers_engine(graph, _ROW), engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(tile_m=128, tile_n=128))


def _sink(h_q):
    return torch.randn(1, h_q, 1, 1, dtype=torch.float32, device="cuda")


def _decode(m, d):
    q = m._bhsd(2, 8, 1, d, torch.float16)
    k, v = (m._bhsd(2, 2, 300, d, torch.float16) for _ in range(2))
    return m._run_dsl_graph(q, k, v, scale=d**-0.5, dtype=torch.float16, sdpa_kwargs={})


def _d128_declines(label):
    return f"SM90's 128 x 128 tile (the D128 template) does not serve {label}"


# The 128 x 128 tile's reason for each feature: config_sm90.d128_decline's label.
_DECLINED_AT_D128 = [
    pytest.param(lambda m, d: m._combo_dense(d, torch.float16, 8, 8, d**-0.5, _sink(8), "causal"), _d128_declines("attention sinks"), id="sink"),
    pytest.param(lambda m, d: m._combo_dense(d, torch.float16, 8, 8, d**-0.5, None, "swa"), _d128_declines("a sliding window"), id="sliding-window"),
    pytest.param(lambda m, d: m._combo_dense(d, torch.float16, 8, 8, d**-0.5, None, "band"), _d128_declines("a widened right band"), id="right-band"),
    pytest.param(lambda m, d: m._combo_dense(d, torch.float16, 8, 8, d**-0.5, None, "padded"), _d128_declines("per-batch lengths"), id="padding-lengths"),
    pytest.param(lambda m, d: m._combo_thd(d, torch.float16, 8, 8, d**-0.5, None, "none"), _d128_declines("THD (ragged) layouts"), id="thd"),
    # A scope decline, not a kernel limit: D128 runs S_q == 1 on the direct route (the b1h2-sq1-skv1 tile case and
    # bottom_right-one_row below); the row keeps the 128 x 128 tile off it because decode is outside the slice #378
    # accepted.
    pytest.param(_decode, "s_q == 1 (decode) is outside the SM90 D128 template's scope", id="s_q-1-scope"),
]


@pytest.mark.L0
@pytest.mark.parametrize("run, why", _DECLINED_AT_D128)
@torch_fork_set_rng(seed=0)
def test_d128_graph_declines_what_d128_does_not_serve(sm100, monkeypatch, run, why):
    """At (128, 128) the row offers the graph on the 64 x 64 tile (the D512 body) and declines the 128 x 128 tile with
    the D128 template's reason, so no D128 plan is ranked for a declaration the template does not serve; at (512, 512)
    the same graph is offered."""
    rank = _ranking(sm100, monkeypatch, _row_verdict)
    offered, reason = rank(lambda: run(sm100, 128))
    assert offered and why in (reason or ""), reason
    assert rank(lambda: run(sm100, 512))[0] is True


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_graph_pinned_pack_gqa_runs_on_the_64_tile(sm100, monkeypatch):
    """The 128 x 128 tile is offered unpacked only: on its MQA graph a strict pin to a packed plan finds the 64 x 64
    one (the D512 body packs), and mismatch() declines PackGQA on the 128 x 128 tile with the D128 template's reason;
    at (512, 512) the same graph has a packed plan."""
    from cudnn.sdpa.fwd import engines

    spec = next(s for s in engines.ENGINE_SPECS if s.name == _ROW)

    def packed(graph):
        try:
            select_engine(graph, _ROW, pack_gqa=True)
            pinned = None
        except AssertionError as exc:
            pinned = str(exc)
        facts = engines.analyze_for(spec, graph)[0]
        return offers_engine(graph, _ROW), pinned, engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(tile_m=128, tile_n=128, pack_gqa=True))

    rank = _ranking(sm100, monkeypatch, packed)

    def run(d):
        q = sm100._bhsd(1, 8, 96, d, torch.float16)
        k, v = (sm100._bhsd(1, 1, 96, d, torch.float16) for _ in range(2))
        return lambda: sm100._run_dsl_graph(q, k, v, scale=d**-0.5, dtype=torch.float16, sdpa_kwargs={}, pack_gqa=True)

    offered, pinned, why = rank(run(128))
    assert offered and pinned is None, pinned
    assert why == _d128_declines("PackGQA"), why
    assert rank(run(512))[:2] == (True, None)


# --- Tiles: whole, partial and single, Q and KV -------
#
# A CTA is 128 query rows (two 64-row warpgroups) by 128 keys, and K and V ride a two-stage ring.
# The shapes cover a Q tile whose second warpgroup has no row (S_q <= 64), a partial row in either
# warpgroup, a KV walk of one tile (fewer tiles than stages), and partial last KV tiles.

_TILE_SHAPES = [
    (2, 4, 128, 128),
    (2, 4, 256, 384),
    (1, 2, 1, 1),
    (1, 2, 2, 63),
    (2, 2, 17, 65),
    (1, 2, 64, 128),
    (1, 2, 65, 129),
    (2, 2, 127, 200),
    (1, 2, 129, 1000),
    (1, 2, 255, 64),
    (1, 1, 1000, 1000),
    (2, 2, 70, 100),
]


@pytest.mark.L0
@pytest.mark.parametrize("i, shape", list(enumerate(_TILE_SHAPES)), ids=[f"b{b}h{h}-sq{sq}-skv{skv}" for b, h, sq, skv in _TILE_SHAPES])
@torch_fork_set_rng(seed=0)
def test_d128_unmasked_tiles_match_the_reference(i, shape):
    """Unmasked MHA; dtype and Stats rotate with the index. (2, 2, 70, 100) is the SM90 suite's
    folded-mask case: an unmasked partial KV tile must keep its padding P a runtime value."""
    b, h, s_q, s_kv = shape
    q, k, v = _inputs(b, h, h, s_q, s_kv, _DTYPES[i % 2])
    o, lse = _run(q, k, v, stats=bool(i // 2 % 2))
    _check(o, lse, q, k, v)


# --- Causal -------

_CAUSAL_CASES = [
    pytest.param(256, 256, False, id="top_left-square"),
    pytest.param(300, 200, False, id="top_left-sq_above_skv"),
    pytest.param(130, 700, True, id="bottom_right-sq_below_skv"),
    pytest.param(129, 129, True, id="bottom_right-square-partial"),
    pytest.param(1, 300, True, id="bottom_right-one_row"),
]


@pytest.mark.L0
@pytest.mark.parametrize("s_q, s_kv, bottom_right", _CAUSAL_CASES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_d128_causal_matches_the_reference(s_q, s_kv, bottom_right, dtype):
    """Causal with Stats: the reverse walk's masked right boundary, unmasked interior and the
    partial-Q-tile arm, for both diagonal anchors."""
    q, k, v = _inputs(2, 4, 4, s_q, s_kv, dtype)
    o, lse = _run(q, k, v, stats=True, is_causal=True, causal_bottom_right=bottom_right)
    # The adapter spells top-left on a dense square as bottom-right (identical there).
    _check(o, lse, q, k, v, is_causal=True, bottom_right=bottom_right or s_q == s_kv)


@pytest.mark.L0
def test_d128_unpadded_bottom_right_above_skv_still_declines():
    """D128 keeps the adapter's pending case: unpadded bottom-right with S_q > S_kv."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    q, k, v = _inputs(1, 2, 2, 200, 100, torch.float16)
    o = torch.empty_like(q)
    with pytest.raises(NotImplementedError, match="unpadded bottom-right S_q > S_kv is not yet supported on SM90"):
        SdpaFwdDslSm90(q, k, v, o, is_causal=True, causal_bottom_right=True).check_support()


# --- GQA and MQA: one KV-head division -------


@pytest.mark.L0
@pytest.mark.parametrize("h_q, h_kv", [(8, 2), (8, 1), (12, 3)], ids=["gqa4", "mqa", "gqa4-h12"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@torch_fork_set_rng(seed=0)
def test_d128_gqa_is_bitwise_repeated_mha(h_q, h_kv, causal):
    """Each CTA does the same arithmetic for GQA as for MHA over repeated K/V heads, so O and
    Stats are equal bit for bit; any difference is a KV-head mapping bug."""
    q, k, v = _inputs(2, h_q, h_kv, 200, 333, torch.bfloat16)
    o, lse = _run(q, k, v, stats=True, is_causal=causal)
    group = h_q // h_kv
    o_mha, lse_mha = _run(q, k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1), stats=True, is_causal=causal)
    assert torch.equal(o, o_mha) and torch.equal(lse, lse_mha)
    _check(o, lse, q, k, v, is_causal=causal)


# --- Stats and scales -------


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_stats_log2_is_the_natural_log_times_log2e():
    """``stats_log2`` scales each natural-log row LSE by log2(e) once, at the store; O is unchanged."""
    q, k, v = _inputs(2, 4, 2, 150, 260, torch.float16)
    o, lse = _run(q, k, v, stats=True, is_causal=True)
    o2, lse2 = _run(q, k, v, stats=True, is_causal=True, stats_log2=True)
    assert torch.equal(o, o2)
    torch.testing.assert_close(lse2, lse * math.log2(math.e), atol=1e-5, rtol=1e-6)


_SCALES = {"negative": -0.25, "zero": 0.0, "unit": 1.0}


# The causal ids are the original ones (kept stable); the unmasked cells are new.
@pytest.mark.L0
@pytest.mark.parametrize(
    "scale, causal",
    [pytest.param(scale, True, id=name) for name, scale in _SCALES.items()]
    + [pytest.param(scale, False, id=f"unmasked-{name}") for name, scale in _SCALES.items()],
)
@torch_fork_set_rng(seed=0)
def test_d128_scale_signs(scale, causal):
    """Each sign compiles its own probability path: a negative scale anchors each row at its
    minimum and always takes the FP64 exponents, a zero scale weighs every visible key equally,
    and a unit scale makes the softmax sharp. With Stats, causal (masked rows) and unmasked: the
    row's zero_scale claim needs both (test_sdpa_graph_analyzer.py::test_zero_attn_scale_claims)."""
    q, k, v = _inputs(2, 4, 2, 130, 200, torch.float16)
    o, lse = _run(q, k, v, stats=True, is_causal=causal, scale_softmax=scale)
    _check(o, lse, q, k, v, scale=scale, is_causal=causal)


@pytest.mark.L0
def test_d128_execute_rejects_operands_that_contradict_the_compiled_plan():
    """The SM90 suite's execute contract on a D128 plan (no sink, which D128 never compiles): no
    launch before compile, Stats presence must match the plan both ways, and the scale's compiled
    sign must match, since a flipped sign is a wrong answer rather than an error."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    q, k, v = _inputs(2, 4, 4, 64, 64, torch.float16)
    o = torch.empty_like(q)
    lse = torch.empty(2, 4, 64, dtype=torch.float32, device="cuda")
    run = lambda api, **kw: api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, **kw)  # noqa: E731

    full = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse)
    assert full.check_support() and full.flavor == (128, 128), full.flavor
    # The refusal under test: execute before compile must raise (the one deliberate execute before compile() here).
    with pytest.raises(RuntimeError, match="requires a compiled plan"):
        run(full, lse_tensor=lse)
    full.compile()
    with pytest.raises(ValueError, match="lse_tensor is required"):
        run(full)
    for wrong in (-1.0, 0.0):
        with pytest.raises(ValueError, match="does not match its compile-time sign mode"):
            run(full, lse_tensor=lse, scale_softmax=wrong)

    plain = SdpaFwdDslSm90(q, k, v, o)
    assert plain.check_support() and plain.flavor == (128, 128), plain.flavor
    plain.compile()
    for kw, why in (
        (dict(lse_tensor=lse), "without an LSE output"),
        (dict(sinks=torch.zeros(1, 4, 1, 1, dtype=torch.float32, device="cuda")), "without sink support"),
        (dict(seq_kv_lens=torch.full((2,), 64, dtype=torch.int32, device="cuda")), "without per-batch KV lengths"),
    ):
        with pytest.raises(ValueError, match=why):
            run(plain, **kw)


# --- Invariance: schedulers, repeats, layouts -------


@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("variant", ["ov", *_PERSISTENT])
@torch_fork_set_rng(seed=0)
def test_d128_schedulers_are_bitwise_equal(variant, causal):
    """NATURAL, LPT and LPT_L2 launch the same tiles in different orders over ~4 waves on 132 SMs;
    each tile's arithmetic is identical, so outputs are equal bit for bit, dense and causal. A persistent variant
    walks the same orders (about four items per CTA), and its NATURAL run equals ov's. The dense persistent record
    is what the adapter ships under every policy: the SCHED_POLICY knob, and the heuristic's LPT past 65535 heads or
    batch."""
    q, k, v = _inputs(4, 16, 4, 1024, 1024, torch.bfloat16)
    mask = dict(is_causal=True) if causal else {}
    runs = [_run(q, k, v, stats=True, sched_policy=policy, variant=variant, **mask) for policy in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2)]
    for o, lse in runs[1:]:
        assert torch.equal(o, runs[0][0]) and torch.equal(lse, runs[0][1])
    if variant != "ov":
        o, lse = _run(q, k, v, stats=True, sched_policy=SCHED_NATURAL, variant="ov", **mask)
        assert torch.equal(o, runs[0][0]) and torch.equal(lse, runs[0][1])
    _check(*runs[0], q, k, v, is_causal=causal)


@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("variant", list(_VARIANTS))
@torch_fork_set_rng(seed=0)
def test_d128_repeat_launches_are_bitwise_equal(variant, causal):
    """Two launches of one plan over the same inputs agree bit for bit (no race, no stale stage), for every
    switch record, dense and causal. A persistent one runs at two CTAs (96 items, 48 per CTA), and both
    launches equal ov's."""
    q, k, v = _inputs(2, 8, 2, 700, 900, torch.float16)
    mask = dict(is_causal=True) if causal else {}
    persistent = variant in _PERSISTENT
    api, o, lse = _plan(q, k, v, stats=True, variant=variant, grid=2 if persistent else None, **mask)
    outs = []
    for _ in range(2):
        o.fill_(float("nan"))
        lse.fill_(float("nan"))
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
        torch.cuda.synchronize()
        outs.append((o.clone(), lse.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    if persistent:
        o_ov, lse_ov = _run(q, k, v, stats=True, variant="ov", **mask)
        assert torch.equal(outs[0][0], o_ov) and torch.equal(outs[0][1], lse_ov)


_LAYOUTS = {
    "bhsd": (0, 1, 2, 3),
    "bshd": (0, 2, 1, 3),
    "sbhd": (2, 0, 1, 3),
}


def _laid_out(t, order):
    """The same BHSD values in storage whose axes run in ``order`` (outermost first)."""
    return t.permute(*order).contiguous().permute(*[order.index(axis) for axis in range(4)])


@pytest.mark.L0
@pytest.mark.parametrize("layout", list(_LAYOUTS), ids=list(_LAYOUTS))
@torch_fork_set_rng(seed=0)
def test_d128_layouts_are_bitwise_equal(layout):
    """Every dense_flex storage order binds natively (the maps carry the strides); the values,
    and so the arithmetic, are the same, so O and Stats equal the BHSD-contiguous run bit for bit."""
    q, k, v = _inputs(2, 4, 2, 100, 140, torch.float16)
    base = _run(*(_laid_out(t, _LAYOUTS["bhsd"]) for t in (q, k, v)), stats=True, is_causal=True)
    other = _run(*(_laid_out(t, _LAYOUTS[layout]) for t in (q, k, v)), stats=True, is_causal=True)
    assert torch.equal(base[0], other[0]) and torch.equal(base[1], other[1])


# --- Overlap and persistence: every switch value computes one function -------
#
# Overlap reorders issues and waits around the same multiplies, P words, V stages and k-step chains, so it equals v1
# bit for bit. Each case is an edge of the overlap protocol.

# K and the even Q rows take this constant: those rows take the FP64 exponents (the large-scores test's max-1.2e12 case),
# while the odd rows (random Q, raw scores about 1e6, under the about 1.3e8 row limit) stay on FP32.
_WIDE = 1.5 * 2.0**16
_CROSS = {  # id: (b, h_q, h_kv, s_q, s_kv, dtype, adapter keywords)
    "ring-wrap": (1, 2, 2, 256, 384, torch.float16, {}),  # 3 KV tiles on the 2-stage rings
    "one-kv-tile": (1, 2, 2, 64, 128, torch.float16, {}),  # prologue and drain only
    "wg2-dead": (2, 2, 2, 17, 300, torch.bfloat16, {}),  # WG2 has no valid row, yet takes every step and releases
    "folded-mask": (2, 2, 2, 70, 100, torch.float16, {}),  # the libNVVM opaque-zero case, now through pack_probability_words
    "partial-q-8kv": (1, 2, 2, 129, 1000, torch.bfloat16, {}),  # 8 KV tiles, all masked (incomplete Q tile: phase 3)
    "causal-br": (2, 4, 4, 130, 700, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),
    "gqa-causal": (2, 8, 2, 200, 333, torch.bfloat16, dict(is_causal=True)),
    "neg-scale": (2, 4, 2, 130, 200, torch.float16, dict(is_causal=True, scale_softmax=-0.25)),  # FP64 path on every tile
    "wide-fp64": (1, 2, 2, 128, 256, torch.bfloat16, {}),  # even rows FP64, odd rows FP32: lanes diverge inside the PV window
    "multi-wave": (4, 16, 4, 1024, 1024, torch.bfloat16, dict(is_causal=True, sched_policy=SCHED_LPT_L2)),
    # For the persistent walk: consecutive items of one CTA differ.
    "alternating-wide": (2, 2, 2, 128, 256, torch.bfloat16, {}),  # head 0 FP64 rows, head 1 FP32: NATURAL items alternate
    "mixed-causal": (1, 2, 2, 640, 640, torch.bfloat16, dict(is_causal=True)),  # Q tiles walk 1 to 5 KV tiles (LPT_L2)
    "one-item": (1, 1, 1, 64, 64, torch.float16, {}),  # one item, one CTA, one KV tile
    "odd-items": (1, 7, 7, 128, 300, torch.float16, {}),  # seven items: at grid 2 one CTA runs four, the other three
    "items-133": (7, 19, 19, 128, 128, torch.bfloat16, {}),  # 132 + 1 items: at the SM count one CTA runs two
    # WG2 live on Q tile 0, dead on Q tile 1 (rows 192..255 >= 129), one KV tile each: at grid 1 WG2's stmatrix follows its
    # previous store by one short item, the smallest window for a deferred store wait wrongly put under the row guard.
    "short-dead-half": (1, 2, 2, 129, 128, torch.float16, {}),
}
# The cases whose inputs _case_inputs fills with _WIDE: their FP64 check is test_d128_large_scores_keep_rows_finite_and_nonzero's.
_WIDE_CASES = ("wide-fp64", "alternating-wide")


def _reference_mask(kw, s_q, s_kv):
    """(is_causal, bottom_right) for _check: the adapter spells top-left on a dense square as bottom-right."""
    causal = kw.get("is_causal", False)
    return causal, causal and (kw.get("causal_bottom_right", False) or s_q == s_kv)


def _case_inputs(case):
    """A ``_CROSS`` case's Q, K and V. ``wide-fp64`` puts the even Q rows and every key on _WIDE; ``alternating-wide``
    puts head 0's Q and K on it, so head 0's rows take the FP64 exponents and head 1's stay on FP32, and a CTA that walks
    consecutive NATURAL items alternates wide and narrow items (an anchor that is not restarted changes bits)."""
    b, h_q, h_kv, s_q, s_kv, dtype, _ = _CROSS[case]
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    if case == "wide-fp64":
        # A row's four lanes share its path, and a warp holds rows of both parities, so its lanes split.
        q[:, :, 0::2].fill_(_WIDE)
        k.fill_(_WIDE)
    elif case == "alternating-wide":
        q[:, 0].fill_(_WIDE)
        k[:, 0].fill_(_WIDE)
    return q, k, v


def _assert_bitwise(variant, case, got, base, against):
    """``got`` (O, Stats) must equal ``base``, the ``against`` variant's run, bit for bit; the failure names the first rows
    that differ and how to tell arithmetic from a race."""
    o, lse = got
    if not (torch.equal(o, base[0]) and torch.equal(lse, base[1])):
        d_o = (o.float() - base[0].float()).abs()
        rows = torch.nonzero(d_o.amax(dim=-1) > 0)[:8].tolist()
        pytest.fail(
            f"{variant} is not {against} on {case}: max |dO| {d_o.max().item():.3e}, Stats equal {torch.equal(lse, base[1])}, "
            f"first differing (batch, head, row) {rows}; rerun test_d128_repeat_launches_are_bitwise_equal[{variant}]: "
            "stable launches mean arithmetic (compare the softmax FFMA/FMUL/FADD counts), moving ones a race"
        )


@pytest.mark.parametrize("case", [pytest.param(c, marks=_level(c in _L0_CROSS)) for c in _CROSS])
@pytest.mark.parametrize("variant", ["ov"])
@torch_fork_set_rng(seed=0)
def test_d128_variant_is_bitwise_v1(variant, case):
    """Every overlap switch value computes v1's O and Stats bit for bit; v1 itself is checked against the
    reference. A difference that repeats launch after launch is arithmetic (compare the softmax block's
    FFMA/FMUL/FADD counts in SASS); one that moves between launches is a race."""
    _, _, _, s_q, s_kv, _, kw = _CROSS[case]
    q, k, v = _case_inputs(case)
    base = _run(q, k, v, stats=True, variant="v1", **kw)
    if case not in _WIDE_CASES:
        is_causal, bottom_right = _reference_mask(kw, s_q, s_kv)
        _check(*base, q, k, v, scale=kw.get("scale_softmax", _SCALE), is_causal=is_causal, bottom_right=bottom_right)
    _assert_bitwise(variant, case, _run(q, k, v, stats=True, variant=variant, **kw), base, "v1")


def _grids(case):
    """The grids worth running for a case: the adapter's (None), then each forced count below the item count (at or
    above it every CTA runs one item, the adapter's grid again)."""
    b, h_q, _, s_q, _, _, _ = _CROSS[case]
    items = math.ceil(s_q / 128) * h_q * b
    return [None] + [g for g in _FORCED_GRIDS if g < items]


# The forced grids are where a cross-item bug shows: _L0_CROSS's cases run every grid in L0, the others at L1.
@pytest.mark.parametrize("case, grid", [pytest.param(c, g, id=f"{c}-grid{g or 'sm'}", marks=_level(c in _L0_CROSS)) for c in _CROSS for g in _grids(c)])
@pytest.mark.parametrize("variant", _PERSISTENT)
@torch_fork_set_rng(seed=0)
def test_d128_persistent_is_bitwise_ov(variant, case, grid):
    """Every persistent record computes the one-tile-per-CTA overlap kernel's O and Stats bit for bit, at the adapter's CTA
    count and at forced ones, where one CTA crosses many item boundaries and the K/V rings and the Q phase wrap at every
    parity. ov is held to v1 by test_d128_variant_is_bitwise_v1."""
    *_, kw = _CROSS[case]
    q, k, v = _case_inputs(case)
    base = _run(q, k, v, stats=True, variant="ov", **kw)
    got = _run(q, k, v, stats=True, variant=variant, grid=grid, **kw)
    _assert_bitwise(variant, f"{case} at grid {grid or 'sm'}", got, base, "ov")


# The walk's rule (causal_walk 1) on causal records. Each case runs at the adapter's CTA count and at forced counts
# below its items, and each forced (case, count) cell is pinned to the branch the plan-time rule picks there
# (_walk_branch), so the cells together take every branch of walk_item and walk_next the shipped constants reach, single
# items only, one item per CTA, and the stride the rule keeps under each deal: below CAUSAL_WALK_MIN_TILES
# (stride-fallback, T 2) and where it balances better (stride-kept, T 9). The walk cases have T >= 9: below that the
# rule keeps the stride, so the branches only T = 1 or 2 reached (middle units alone, middle tiles with light rows) run
# only with CAUSAL_WALK_MIN_TILES at -1 (a source edit; test_paired_walk_visits_every_item_once_in_the_dealt_order
# decodes them on plain ints). A plan of more items than the GPU has SMs walks the NATURAL order check_support gives it;
# the others are routed one tile per CTA by check_support and take the causal order (LPT_L2 here) into the forced
# record. bottom_right is the template's (the adapter spells top-left on a square as bottom-right).
_WALK_CASES = {  # id: (b, h_q, h_kv, s_q, s_kv, dtype, adapter keywords)
    "odd-t": (4, 4, 4, 1152, 1152, torch.bfloat16, dict(is_causal=True)),  # T = 9: middle units
    "odd-groups": (3, 5, 5, 1152, 1152, torch.float16, dict(is_causal=True, causal_bottom_right=True)),  # a lone middle
    "sq-above-skv": (2, 4, 4, 1408, 1024, torch.bfloat16, dict(is_causal=True)),  # top-left: rows past S_kv see every key
    "diag-576": (2, 4, 2, 1152, 1728, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),  # not a tile multiple
    "partial-ragged": (2, 4, 2, 1100, 1200, torch.float16, dict(is_causal=True, causal_bottom_right=True)),
    "pairs-t10": (1, 1, 1, 1280, 1280, torch.bfloat16, dict(is_causal=True)),  # T = 10: pairs alone on one CTA
    "gqa2-t10": (1, 2, 1, 1280, 1280, torch.bfloat16, dict(is_causal=True)),
    "middle-light-lone": (1, 1, 1, 1152, 768, torch.float16, dict(is_causal=True)),  # a middle unit, light rows, a lone one
    "gqa8": (2, 8, 1, 1152, 1152, torch.bfloat16, dict(is_causal=True)),
    "items-200": (5, 5, 5, 1152, 1152, torch.bfloat16, dict(is_causal=True)),  # 225 items: two rounds at the SM count
    "stride-fallback": (8, 24, 8, 256, 2048, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),  # chunked, T 2
    "stride-kept": (4, 8, 8, 1152, 5248, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),  # chunked, T 9
    "mixed": (1, 2, 2, 1152, 1152, torch.bfloat16, dict(is_causal=True)),
    "multi-wave": (4, 16, 4, 1152, 1152, torch.bfloat16, dict(is_causal=True)),
}
_WALK_CELLS = {  # (case, forced CTA count): the branch there (_walk_branch)
    ("odd-t", 1): "pairs-middles",
    ("odd-t", 2): "pairs-middles",
    ("odd-t", 3): "pairs-middles",
    ("odd-t", 131): "singles",
    ("odd-groups", 1): "pairs-middles+lone",
    ("odd-groups", 2): "pairs-middles+light+lone",
    ("odd-groups", 3): "pairs+light",
    ("odd-groups", 131): "singles",
    ("sq-above-skv", 1): "pairs-middles",
    ("sq-above-skv", 2): "pairs-middles",
    ("sq-above-skv", 3): "pairs+light",
    ("diag-576", 1): "pairs-middles",
    ("diag-576", 2): "pairs-middles",
    ("diag-576", 3): "pairs-middles",
    ("partial-ragged", 1): "pairs-middles",
    ("partial-ragged", 2): "pairs-middles",
    ("partial-ragged", 3): "pairs-middles",
    ("pairs-t10", 1): "pairs",
    ("pairs-t10", 2): "pairs+light",
    ("pairs-t10", 3): "pairs+light",
    ("gqa2-t10", 1): "pairs",
    ("gqa2-t10", 2): "pairs",
    ("gqa2-t10", 3): "pairs-middles+light",
    ("middle-light-lone", 1): "pairs-middles+lone",
    ("middle-light-lone", 2): "pairs+light",
    ("middle-light-lone", 3): "middles+light+lone",
    ("gqa8", 1): "pairs-middles",
    ("gqa8", 2): "pairs-middles",
    ("gqa8", 3): "pairs-middles",
    ("gqa8", 131): "singles",
    ("items-200", 1): "pairs-middles+lone",
    ("items-200", 2): "pairs+light",
    ("items-200", 3): "pairs+light",
    ("items-200", 131): "singles",
    ("stride-fallback", 1): "stride",
    ("stride-fallback", 2): "stride-snake",
    ("stride-fallback", 3): "stride",
    ("stride-fallback", 131): "stride",
    ("stride-kept", 1): "pairs-middles",
    ("stride-kept", 2): "pairs-middles",
    ("stride-kept", 3): "pairs-middles",
    ("stride-kept", 131): "stride-snake",
    ("mixed", 1): "pairs-middles",
    ("mixed", 2): "pairs+light",
    ("mixed", 3): "pairs-middles",
    ("multi-wave", 1): "pairs-middles",
    ("multi-wave", 2): "pairs-middles",
    ("multi-wave", 3): "pairs-middles",
    ("multi-wave", 131): "pairs+light",
}


def _walk_branch(api, grid):
    """The branch a compiled plan of the walk's rule takes at ``grid`` CTAs (None: the adapter's count), from the plan's
    own rule (the module's causal_walk_plan over its record): which unit kinds its walk has, or "stride" and
    "stride-snake" for the stride the rule keeps under its deal (causal_stride_snake)."""
    p = api.params
    ctas = grid or api._persistent_ctas(api.q_desc.device)
    shape = (api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max)
    args = (shape, p.causal if p.bottom_right is None else p.bottom_right, p.sched_policy, p.qh_per_kh, ctas, 2)
    walk = api._k_mod.causal_walk_plan(*args)
    if walk is None:
        return "stride-snake" if api._k_mod.causal_stride_snake(*args) else "stride"
    if walk.heavy_units == 0:
        return "singles" if walk.units > ctas else "one-per-cta"
    kind = "pairs" if not walk.middles else "middles" if walk.pairs == 0 else "pairs-middles"
    return kind + ("+light" if walk.light_rows else "") + ("+lone" if walk.lone_middle else "")


def _walk_grids(case):
    b, h_q, _, s_q, _, _, _ = _WALK_CASES[case]
    items = math.ceil(s_q / 128) * h_q * b
    return [None] + [g for g in _FORCED_GRIDS if g < items]


@pytest.mark.L0
def test_d128_paired_walk_cells_cover_every_branch():
    """The pinned cells reach every branch of the paired walk the shipped constants reach (T >= CAUSAL_WALK_MIN_TILES): pairs, pairs with
    middle units, middle tiles only (T - light_rows of 1) with light rows and a lone middle, light rows and a lone
    middle with the others, single items only, and the stride the rule keeps under each deal; and every forced count
    below a case's items is pinned."""
    assert {key for key in _WALK_CELLS} == {(c, g) for c in _WALK_CASES for g in _walk_grids(c) if g}, "a forced cell without its pin"
    kinds = set(_WALK_CELLS.values())
    assert {"pairs", "pairs+light", "pairs-middles", "pairs-middles+light", "pairs-middles+lone", "pairs-middles+light+lone"} <= kinds, kinds
    assert {"middles+light+lone", "singles", "stride", "stride-snake"} <= kinds, kinds
    from cudnn.sdpa.fwd.config_sm90 import CAUSAL_WALK_MIN_TILES

    walks = [case for (case, _), kind in _WALK_CELLS.items() if not kind.startswith("stride")]
    assert all(math.ceil(_WALK_CASES[case][3] / 128) >= CAUSAL_WALK_MIN_TILES for case in walks), "a walk cell below CAUSAL_WALK_MIN_TILES Q tiles"


@pytest.mark.parametrize(
    "case, grid", [pytest.param(c, g, id=f"{c}-grid{g or 'sm'}", marks=_level(c in _L0_WALK)) for c in _WALK_CASES for g in _walk_grids(c)]
)
@torch_fork_set_rng(seed=0)
def test_d128_paired_walk_is_bitwise_ov(case, grid):
    """A paired-walk record computes the one-tile-per-CTA overlap kernel's O and Stats bit for bit, on causal cases at
    every walk branch (_WALK_CELLS) and at the adapter's CTA count, where one CTA runs pairs, middles and light rows in
    turn and the K/V rings and the Q phase wrap at every parity across items of different lengths."""
    b, h_q, h_kv, s_q, s_kv, dtype, kw = _WALK_CASES[case]
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    base = _run(q, k, v, stats=True, variant="ov", **kw)
    api, o, lse = _plan(q, k, v, stats=True, variant="pw", grid=grid, **kw)
    if grid:
        assert _walk_branch(api, grid) == _WALK_CELLS[(case, grid)], (case, grid, _walk_branch(api, grid))
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    torch.cuda.synchronize()
    _assert_bitwise("pw", f"{case} at grid {grid or 'sm'}", (o, lse), base, "ov")


# The NATURAL-order stride a causal record of the walk's rule keeps (causal_walk 1; sched_policy NATURAL pinned, so the
# cases the adapter routes one tile per CTA keep that order in the forced record too) under each deal. Every cell is
# pinned to the deal causal_stride_snake picks there (_walk_branch: "stride" deals each round from CTA 0,
# "stride-snake" deals odd rounds in reverse): T = 1, 2, 3, 8, 9 and 17, top-left and bottom-right, MHA, GQA and MQA.
# Below CAUSAL_WALK_MIN_TILES Q tiles every plan is the stride; at T 9 and 17 the rule keeps it only where it balances
# better, on chunked prefill bottom-right here (on no top-left shape searched at T >= 9 did it keep the stride). At T 1
# every item does the same KV work, so the snake is never strictly lighter and both T 1 cases deal straight.
_STRIDE_CASES = {  # id: (b, h_q, h_kv, s_q, s_kv, dtype, adapter keywords)
    "t1-tl-mha": (2, 4, 4, 100, 300, torch.float16, dict(is_causal=True)),
    "t1-br-gqa": (2, 8, 2, 100, 300, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),
    "t2-tl-gqa": (1, 8, 2, 256, 512, torch.bfloat16, dict(is_causal=True)),
    "t2-br-mqa": (1, 4, 1, 256, 512, torch.float16, dict(is_causal=True, causal_bottom_right=True)),
    "t3-tl-mqa": (1, 4, 1, 384, 640, torch.bfloat16, dict(is_causal=True)),
    "t3-br-mha": (1, 4, 4, 384, 640, torch.float16, dict(is_causal=True, causal_bottom_right=True)),
    "t8-tl-mha": (1, 4, 4, 1024, 1280, torch.bfloat16, dict(is_causal=True)),
    "t8-br-gqa": (1, 8, 2, 1024, 1280, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),
    "t9-br-mha": (3, 4, 4, 1152, 9344, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),
    "t9-br-mqa": (12, 4, 1, 1152, 9344, torch.float16, dict(is_causal=True, causal_bottom_right=True)),
    "t17-br-mha": (3, 4, 4, 2176, 10368, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),
    "t17-br-gqa": (4, 8, 2, 2148, 10340, torch.float16, dict(is_causal=True, causal_bottom_right=True)),  # partial Q tile
}
_STRIDE_CELLS = {  # (case, forced CTA count): the deal there (_walk_branch)
    ("t1-tl-mha", 2): "stride",
    ("t1-tl-mha", 3): "stride",
    ("t1-br-gqa", 2): "stride",
    ("t1-br-gqa", 3): "stride",
    ("t2-tl-gqa", 2): "stride-snake",
    ("t2-tl-gqa", 3): "stride",
    ("t2-br-mqa", 2): "stride-snake",
    ("t2-br-mqa", 3): "stride",
    ("t3-tl-mqa", 2): "stride",
    ("t3-tl-mqa", 3): "stride-snake",
    ("t3-br-mha", 2): "stride",
    ("t3-br-mha", 3): "stride-snake",
    ("t8-tl-mha", 2): "stride-snake",
    ("t8-tl-mha", 3): "stride",
    ("t8-br-gqa", 2): "stride-snake",
    ("t8-br-gqa", 5): "stride",
    ("t9-br-mha", 10): "stride",
    ("t9-br-mha", 12): "stride-snake",
    ("t9-br-mqa", 29): "stride",
    ("t9-br-mqa", 30): "stride-snake",
    ("t17-br-mha", 7): "stride-snake",
    ("t17-br-mha", 9): "stride",
    ("t17-br-gqa", 78): "stride",
    ("t17-br-gqa", 111): "stride-snake",
}


@pytest.mark.L0
def test_d128_natural_stride_cells_cover_each_deal():
    """The NATURAL-stride cells take T = 1, 2, 3, 8, 9 and 17, each T above 1 under both deals, and each mask and each
    head layout under both deals (T 9 and 17 bottom-right only); every cell's count is below its case's items."""
    tiles = {case: math.ceil(c[3] / 128) for case, c in _STRIDE_CASES.items()}
    assert set(tiles.values()) == {1, 2, 3, 8, 9, 17}, tiles
    assert {case for case, _ in _STRIDE_CELLS} == set(_STRIDE_CASES)
    for (case, grid), deal in _STRIDE_CELLS.items():
        b, h_q, *_ = _STRIDE_CASES[case]
        assert deal in ("stride", "stride-snake") and grid < tiles[case] * h_q * b, (case, grid)
    deals = lambda pick: {deal for (case, _), deal in _STRIDE_CELLS.items() if pick(case)}
    for t in (2, 3, 8, 9, 17):
        assert deals(lambda case: tiles[case] == t) == {"stride", "stride-snake"}, t
    for word in ("-tl-", "-br-", "-mha", "-gqa", "-mqa"):
        assert deals(lambda case: word in case) == {"stride", "stride-snake"}, word
    layouts = {"-mha": lambda h_q, h_kv: h_q == h_kv, "-gqa": lambda h_q, h_kv: 1 < h_kv < h_q, "-mqa": lambda h_q, h_kv: h_kv == 1}
    for case, (_, h_q, h_kv, *_rest) in _STRIDE_CASES.items():
        assert sum(layouts[word](h_q, h_kv) for word in layouts if word in case) == 1, case
        assert ("causal_bottom_right" in _rest[-1]) is ("-br-" in case), case


@pytest.mark.parametrize("case, grid", [pytest.param(c, g, id=f"{c}-grid{g}", marks=_level((c, g) in _L0_STRIDE)) for c, g in _STRIDE_CELLS])
@torch_fork_set_rng(seed=0)
def test_d128_natural_stride_is_bitwise_ov(case, grid):
    """A causal record of the walk's rule on its NATURAL-order stride computes the one-tile-per-CTA overlap kernel's O
    and Stats bit for bit under each deal (_STRIDE_CELLS): a CTA's items come from different Q tiles and groups each
    round, and under the snake deal the order within a round flips every other round, so the K/V rings and the Q phase
    wrap across items of different lengths in both directions."""
    b, h_q, h_kv, s_q, s_kv, dtype, kw = _STRIDE_CASES[case]
    kw = dict(kw, sched_policy=SCHED_NATURAL)
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    base = _run(q, k, v, stats=True, variant="ov", **kw)
    api, o, lse = _plan(q, k, v, stats=True, variant="pw", grid=grid, **kw)
    assert api.params.sched_policy == SCHED_NATURAL, api.params
    assert _walk_branch(api, grid) == _STRIDE_CELLS[(case, grid)], (case, grid, _walk_branch(api, grid))
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
    torch.cuda.synchronize()
    _assert_bitwise("pw", f"{case} NATURAL stride at grid {grid}", (o, lse), base, "ov")


@pytest.mark.L0
@pytest.mark.parametrize("grid", [2, 3], ids=["grid2", "grid3"])
@torch_fork_set_rng(seed=0)
def test_d128_paired_walk_repeats_and_replays_bitwise(grid):
    """Two launches of one paired-walk plan agree bit for bit and equal ov's, and its CUDA-graph replay equals its
    eager launch before and after the inputs change in place (bottom-right causal, partial Q and ragged KV, T 10:
    pairs alone at 2 CTAs, pairs then light rows at 3)."""
    q, k, v = _inputs(1, 4, 2, 1175, 1275, torch.bfloat16)
    mask = dict(is_causal=True, causal_bottom_right=True)
    api, o, lse = _plan(q, k, v, stats=True, variant="pw", grid=grid, **mask)
    assert _walk_branch(api, grid) == {2: "pairs", 3: "pairs+light"}[grid], (grid, _walk_branch(api, grid))

    def run():
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)

    outs = []
    for _ in range(2):
        o.fill_(float("nan"))
        lse.fill_(float("nan"))
        run()
        torch.cuda.synchronize()
        outs.append((o.clone(), lse.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    _assert_bitwise("pw", f"repeat at grid {grid}", outs[0], _run(q, k, v, stats=True, variant="ov", **mask), "ov")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            run()
            with torch.cuda.graph(graph, stream=stream):
                run()
        stream.synchronize()
        for step in range(2):
            if step:
                k.mul_(0.5)
                v.add_(1.0)
            run()
            torch.cuda.synchronize()
            eager = (o.clone(), lse.clone())
            o.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(o, eager[0]) and torch.equal(lse, eager[1]), (grid, "changed inputs" if step else "same inputs")
    finally:
        graph.reset()


# The claim counter (TemplateParams.causal_claim): every item after a CTA's first comes from the Int32 counter at the
# caller's workspace base, which the native dense binder zeroes on the launch stream before every launch. The cases take
# causal top-left and bottom-right, MHA, GQA and MQA, and T 1, 2, 6, 8, 9, 17 and 32; each runs at the adapter's CTA
# count (where only t17-br-mqa, 136 items, has more items than an H100's SMs) and at forced counts far below its items,
# where one CTA claims many items of different lengths. Sanitizer recipe for the dev session, one cell each:
#   compute-sanitizer --tool racecheck python -m pytest -q -p no:cacheprovider test_sdpa_fwd_d128_sm90.py -k "claim_counter_is_bitwise_ov and pc1 and t8-br-gqa-grid3"
#   compute-sanitizer --tool synccheck (same selection); compute-sanitizer --tool memcheck (same selection)
_CLAIM_CASES = {  # id: (b, h_q, h_kv, s_q, s_kv, dtype, adapter keywords)
    "t1-br-gqa": (2, 8, 2, 100, 300, torch.float16, dict(is_causal=True, causal_bottom_right=True)),
    "t2-tl-mqa": (1, 4, 1, 256, 512, torch.bfloat16, dict(is_causal=True)),
    "t6-tl-mha": (2, 4, 4, 768, 768, torch.bfloat16, dict(is_causal=True)),
    "t8-br-gqa": (1, 8, 2, 1024, 1280, torch.bfloat16, dict(is_causal=True, causal_bottom_right=True)),
    "t9-tl-mha": (1, 4, 4, 1152, 1152, torch.float16, dict(is_causal=True)),
    "t17-br-mqa": (2, 4, 1, 2148, 2300, torch.float16, dict(is_causal=True, causal_bottom_right=True)),  # partial Q tile
    "t32-tl-gqa": (1, 4, 2, 4096, 4096, torch.bfloat16, dict(is_causal=True)),
}
_CLAIM_GRIDS = (None, 1, 3)


def _claim_items(case):
    b, h_q, _, s_q, *_ = _CLAIM_CASES[case]
    return math.ceil(s_q / 128) * h_q * b


def _poisoned_workspace(nbytes=256):
    """A caller workspace whose counter word reads 0x7F7F7F7F: the binder's reset must zero it before the launch."""
    return torch.full((nbytes,), 0x7F, dtype=torch.uint8, device="cuda")


def _counter(workspace):
    """The claim counter's word after a launch: the item count (every CTA's claims, its one end marker included)."""
    return int(workspace[:4].view(torch.int32).item())


def _claim_plan(q, k, v, variant, *, grid=None, **kw):
    """A claim record (``variant`` in _CLAIM_VARIANTS) over NaN-filled O and Stats, with NATURAL pinned (the counter
    numbers its items itself; validate_params refuses it on another order). It reserves the counter's 128 bytes and its
    prepared spec carries the binder's flag."""
    api, o, lse = _plan(q, k, v, stats=True, variant=variant, grid=grid, **dict(kw, sched_policy=SCHED_NATURAL))
    assert api.params.causal_claim == _CLAIM_VARIANTS[variant] and api._k_mod.CAUSAL_CLAIM == _CLAIM_VARIANTS[variant]
    assert api.scratch_workspace_bytes() == 128
    assert api._dense_spec.claim_counter
    return api, o, lse


def _claim_run(api, q, k, v, o, lse, workspace):
    o.fill_(float("nan"))
    lse.fill_(float("nan"))
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, workspace=workspace)
    torch.cuda.synchronize()
    return o.clone(), lse.clone()


@pytest.mark.L0
def test_d128_claim_counter_cases_take_every_t_mask_and_head_layout():
    """The claim cases take T = 1, 2, 6, 8, 9, 17 and 32, both masks, MHA, GQA and MQA, a partial Q tile, and one case
    with more items than an H100's 132 SMs at the adapter's count; every forced count is below its case's items."""
    tiles = {case: math.ceil(c[3] / 128) for case, c in _CLAIM_CASES.items()}
    assert set(tiles.values()) == {1, 2, 6, 8, 9, 17, 32}, tiles
    layouts = {"-mha": lambda h_q, h_kv: h_q == h_kv, "-gqa": lambda h_q, h_kv: 1 < h_kv < h_q, "-mqa": lambda h_q, h_kv: h_kv == 1}
    for case, (_, h_q, h_kv, s_q, _, _, kw) in _CLAIM_CASES.items():
        assert sum(layouts[word](h_q, h_kv) for word in layouts if word in case) == 1, case
        assert kw.get("causal_bottom_right", False) is ("-br-" in case), case
        assert all(grid < _claim_items(case) for grid in _CLAIM_GRIDS if grid), case
    assert {word for word in ("-tl-", "-br-") if any(word in case for case in _CLAIM_CASES)} == {"-tl-", "-br-"}
    assert any(_claim_items(case) > 132 for case in _CLAIM_CASES) and any(c[3] % 128 for c in _CLAIM_CASES.values())


@pytest.mark.parametrize(
    "case, grid", [pytest.param(c, g, id=f"{c}-grid{g or 'sm'}", marks=_level(c in _L0_CLAIM)) for c in _CLAIM_CASES for g in _CLAIM_GRIDS]
)
@pytest.mark.parametrize("variant", list(_CLAIM_VARIANTS))
@torch_fork_set_rng(seed=0)
def test_d128_claim_counter_is_bitwise_ov(variant, case, grid):
    """A claim record computes the one-tile-per-CTA overlap kernel's O and Stats bit for bit from a poisoned workspace,
    with a group's Q tiles in NATURAL order, and leaves the counter at the item count: every item ran once, and every
    CTA took its one end marker."""
    b, h_q, h_kv, s_q, s_kv, dtype, kw = _CLAIM_CASES[case]
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    base = _run(q, k, v, stats=True, variant="ov", **kw)
    api, o, lse = _claim_plan(q, k, v, variant, grid=grid, **kw)
    workspace = _poisoned_workspace()
    got = _claim_run(api, q, k, v, o, lse, workspace)
    _assert_bitwise(variant, f"{case} at grid {grid or 'sm'}", got, base, "ov")
    assert _counter(workspace) == _claim_items(case), (case, grid, _counter(workspace))


@pytest.mark.L0
@pytest.mark.parametrize("variant", list(_CLAIM_VARIANTS))
@torch_fork_set_rng(seed=0)
def test_d128_claim_counter_resets_repeats_and_replays(variant):
    """The binder zeroes the counter before every launch, so a claim record gives ov's bits from 0x7F bytes, from a
    stale count and from the previous launch's count, launch after launch; and a captured graph zeroes it on every
    replay (a memset node before the kernel), after the workspace is refilled with 0x7F, on a second graph captured
    over another workspace pointer too (bottom-right causal, partial Q tile and ragged KV, GQA 2, at 3 CTAs)."""
    q, k, v = _inputs(1, 4, 2, 1175, 1275, torch.bfloat16)
    mask = dict(is_causal=True, causal_bottom_right=True)
    items = math.ceil(1175 / 128) * 4
    base = _run(q, k, v, stats=True, variant="ov", **mask)
    api, o, lse = _claim_plan(q, k, v, variant, grid=3, **mask)
    ws_a, ws_b = _poisoned_workspace(), _poisoned_workspace(512)
    for start in (None, items // 2, items, -1):  # 0x7F bytes, a stale count, the previous launch's count, all ones
        if start is not None:
            ws_a[:4].view(torch.int32).fill_(start)
        _assert_bitwise(variant, f"eager from {start}", _claim_run(api, q, k, v, o, lse, ws_a), base, "ov")
        assert _counter(ws_a) == items, (start, _counter(ws_a))

    def run(workspace):
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, workspace=workspace)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = (torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph())
    try:
        with torch.cuda.stream(stream):
            run(ws_a)
            with torch.cuda.graph(graphs[0], stream=stream):
                run(ws_a)
            with torch.cuda.graph(graphs[1], stream=stream):
                run(ws_b)
        stream.synchronize()
        for name, graph, workspace in (("A", graphs[0], ws_a), ("B", graphs[1], ws_b)):
            for step in range(2):
                workspace.fill_(0x7F)
                o.fill_(float("nan"))
                lse.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                _assert_bitwise(variant, f"replay {step} over workspace {name}", (o, lse), base, "ov")
                assert _counter(workspace) == items, (name, step, _counter(workspace))
    finally:
        for graph in graphs:
            graph.reset()


# The claim ring's warm stress (Rule 11's run-time detector): t32-tl-gqa, 128 items, at 16 forced CTAs, so each CTA
# claims about eight items and refills the ring's two slots about six times in every launch (KV tiles per Q tile 1
# to 32).
_CLAIM_STRESS = ("t32-tl-gqa", 16)


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=0)
def test_d128_claim_ring_warm_stress():
    """python/cudnn/AGENTS.md Rule 11 and python/cudnn/sdpa/AGENTS.md S15 at run time, for the claim counter's id ring:
    a claim record whose CTAs each publish several ids, so the producer refills each slot inside one launch,
    launched back to back and behind a shared-memory poison launch (warm_stress.assert_warm_stress), every O and Stats
    byte compared with the one-tile-per-CTA overlap kernel's (ov, which has no id ring), and the counter left at the
    item count. A zero is a rate bound on this GPU behind the positive control, and the kernel runs one thread block
    per SM (its shared memory), which the report flags; the proof of the order is the rendering pin
    (test_d128_register_split_no_spills_and_no_wgmma_serialization[pc1-causal])."""
    from warm_stress import StressCase, assert_warm_stress

    case, grid = _CLAIM_STRESS
    b, h_q, h_kv, s_q, s_kv, dtype, kw = _CLAIM_CASES[case]
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    base = _run(q, k, v, stats=True, variant="ov", **kw)
    api, o, lse = _claim_plan(q, k, v, "pc1", grid=grid, **kw)
    workspace = _poisoned_workspace()
    items = _claim_items(case)
    # A CTA publishes one id per item it runs (the next item's, the last its end marker); its first two fill the two
    # fresh slots, every later one refills a slot the compute warps released.
    refills = items - 2 * grid
    assert refills > 0, (items, grid)

    def launch():
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, workspace=workspace)

    _assert_bitwise("pc1", f"{case} at grid {grid}", _claim_run(api, q, k, v, o, lse, workspace), base, "ov")
    assert _counter(workspace) == items, (case, grid, _counter(workspace))
    assert_warm_stress(
        StressCase(
            f"SM90 D128 claim counter id ring ({case}, {grid} CTAs)",
            launch,
            {"o_storage": o.transpose(1, 2), "stats": lse},
            thread_blocks=grid,
            refills_per_launch=refills,
            reference={"o_storage": base[0].transpose(1, 2).contiguous(), "stats": base[1].contiguous()},
        ),
        warm=256,
        poisoned=256,
    )
    assert _counter(workspace) == items, (case, grid, _counter(workspace))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_claim_counter_workspace_is_required_and_checked_before_launch():
    """A claim record refuses a missing, short, misaligned or host workspace, and one that overlaps an operand (the
    binder: the counter word would be zeroed under O), each before any launch: O keeps its NaN fill."""
    q, k, v = _inputs(1, 4, 2, 1024, 1024, torch.bfloat16)
    api, o, lse = _claim_plan(q, k, v, "pc1", grid=3, is_causal=True)
    spare = torch.empty(512, dtype=torch.uint8, device="cuda")
    o_storage = o.transpose(1, 2)  # O's BSHD storage under its BHSD view
    assert o_storage.is_contiguous()
    cases = [
        (None, "requires a 128-byte workspace"),
        (spare[:64], "requires a 128-byte workspace; the provided buffer has 64 bytes"),
        (spare[4:260], "16-byte aligned"),
        (torch.empty(256, dtype=torch.uint8), "on the Q tensor's CUDA device"),
        (o_storage.reshape(-1).view(torch.uint8)[:256], "workspace overlaps o$"),
    ]
    for workspace, why in cases:
        o.fill_(float("nan"))
        with pytest.raises(ValueError, match=why):
            api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, workspace=workspace)
        torch.cuda.synchronize()
        assert torch.isnan(o).all(), f"a refused launch ({why}) wrote O"


@pytest.mark.L0
def test_d128_adapter_gives_the_claim_counter_only_past_its_gate(monkeypatch):
    """check_support gives the claim record (persistent, the walk off, causal_claim D128_CAUSAL_CLAIM, NATURAL, 128
    bytes of workspace) only to a causal record of the walk's rule the route keeps, in NATURAL order, with more items
    than D128_CLAIM_MIN_ROUNDS x CTAs (the switch shipped off, set on here; the gate 93 rounds); never on a routed or
    dense plan or a pinned LPT_L2 order, nor with the switch at 0, and an out of range switch (2, the heaviest-first
    numbering s29 dropped, or 3) raises. The CTA count is patched to 132 so the gate does not move with the board."""
    from cudnn.sdpa.fwd import api_dsl

    def record(shape, mask, **kw):
        q, k, v = _inputs(*shape, torch.bfloat16)
        api = api_dsl.SdpaFwdDslSm90(q, k, v, torch.empty_like(q), **mask, **kw)
        api._persistent_ctas = lambda device: 132
        assert api.check_support()
        return api

    kept, routed, causal = (4, 16, 4, 1152, 1152), (1, 4, 4, 1024, 1024), dict(is_causal=True)  # 576 items L 5; 32 items
    assert (api_dsl.D128_CAUSAL_CLAIM, api_dsl.D128_CLAIM_MIN_ROUNDS) == (0, 93)
    assert record(kept, causal).params.causal_claim == 0, "the switch ships off"
    monkeypatch.setattr(api_dsl, "D128_CAUSAL_CLAIM", 1)
    assert record(kept, causal).params.causal_claim == 0, "576 items is below the shipped gate's 93 x 132"
    monkeypatch.setattr(api_dsl, "D128_CLAIM_MIN_ROUNDS", 4)  # 576 > 528
    api = record(kept, causal)
    p = api.params
    assert (p.persistent, p.causal_walk, p.causal_claim, p.sched_policy) == (True, 0, 1, SCHED_NATURAL), p
    assert api.scratch_workspace_bytes() == 128
    monkeypatch.setattr(api_dsl, "D128_CAUSAL_CLAIM", 0)
    assert record(kept, causal).params.causal_claim == 0 and record(kept, causal).scratch_workspace_bytes() == 0
    monkeypatch.setattr(api_dsl, "D128_CAUSAL_CLAIM", 1)
    monkeypatch.setattr(api_dsl, "D128_CLAIM_MIN_ROUNDS", 5)  # 576 <= 660
    assert record(kept, causal).params.causal_claim == 0
    monkeypatch.setattr(api_dsl, "D128_CLAIM_MIN_ROUNDS", 0)
    routed_params = record(routed, causal).params
    assert (routed_params.persistent, routed_params.causal_claim) == (False, 0), routed_params
    assert record(kept, causal, sched_policy=SCHED_LPT_L2).params.causal_claim == 0
    assert record(kept, {}).params.causal_claim == 0 and record(kept, {}).scratch_workspace_bytes() == 0
    for bad in (2, 3):
        monkeypatch.setattr(api_dsl, "D128_CAUSAL_CLAIM", bad)
        with pytest.raises(ValueError, match="D128_CAUSAL_CLAIM must be 0 or 1"):
            record(kept, causal)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_graph_reports_and_binds_the_claim_counter_workspace(graph_d128, monkeypatch):
    """Through the graph route a claim record reports its 128 bytes (graph.get_workspace_size), keeps the ordered
    native path (the binder's ordered_workspace_bytes matches), and computes O from the runner's uninitialized workspace
    (b4 h8/8 S 1152 causal: 288 items, L 5, the walk's record the route keeps, at the counter's gate patched to 0)."""
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D128_CAUSAL_CLAIM", 1)
    monkeypatch.setattr(api_dsl, "D128_CLAIM_MIN_ROUNDS", 0)
    graphs, pin = [], graph_d128._select_engine
    monkeypatch.setattr(graph_d128, "_select_engine", lambda graph, name, **kw: graphs.append(graph) or pin(graph, name, **kw))
    q, k, v = _inputs(4, 8, 8, 1152, 1152, torch.bfloat16)
    o = graph_d128._run_dsl_graph(q, k, v, scale=_SCALE, dtype=torch.bfloat16, sdpa_kwargs=dict(use_causal_mask=True))
    torch.testing.assert_close(o, _ref_sdpa_full(q, k, v, scale=_SCALE, is_causal=True), atol=5e-2, rtol=3e-2)
    (graph,) = graphs
    plan = graph._compiled_plans[graph._plan_index]
    assert plan._compiled.kernel_module.CAUSAL_CLAIM == 1
    assert graph.get_workspace_size() >= 128 and plan.get_workspace_size() == 128
    assert plan._ordered_native is not None, "the claim plan lost the ordered native path"


@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("variant", list(_VARIANTS))
@torch_fork_set_rng(seed=0)
def test_d128_graph_replay_is_bitwise_eager(variant, causal):
    """CUDA-graph replay of one plan equals its eager launch bit for bit, before and after the inputs change
    in place at the captured pointers, for every switch record, dense and bottom-right causal; a
    persistent one at two CTAs (24 items, 12 per CTA)."""
    q, k, v = _inputs(2, 4, 2, 300, 700, torch.bfloat16)
    mask = dict(is_causal=True, causal_bottom_right=True) if causal else {}
    api, o, lse = _plan(q, k, v, stats=True, variant=variant, grid=2 if variant in _PERSISTENT else None, **mask)

    def run():
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        # Warm up on the capture stream, as test_sdpa_native_sm90_binding.py's capture does.
        with torch.cuda.stream(stream):
            run()
            with torch.cuda.graph(graph, stream=stream):
                run()
        stream.synchronize()
        for step in range(2):
            if step:
                k.mul_(0.5)
                v.add_(1.0)
            run()
            torch.cuda.synchronize()
            eager = (o.clone(), lse.clone())
            o.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(o, eager[0]) and torch.equal(lse, eager[1]), (variant, causal, "changed inputs" if step else "same inputs")
    finally:
        graph.reset()


# Compiles one plan per (CTA count, case), then runs each three times under a faulthandler deadline: a ring that never
# completes makes the child exit instead of hanging the worker (test/AGENTS.md, crash isolation), and the parent reports
# HANG. Self-contained, so it needs nothing from pytest's path setup.
_TURNS_CHILD = textwrap.dedent(r"""
    import dataclasses, faulthandler, json, sys, time
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    switches, run_budget_s, counts = json.loads(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3])
    # (b, h_q, h_kv, s_q, s_kv, causal, bottom_right): ring wrap; one KV tile (the prologue meets the drain); WG2 without
    # a row over three tiles of an incomplete Q tile (phase 3); bottom-right causal with masked and unmasked phases (16
    # items, eight per CTA at count 2); top-left causal square, whose Q tiles walk one and two KV tiles, so one CTA
    # releases Q from the first tile and from the last step in turn. The paired walk takes pairs alone on those two, so
    # two more: a top-left square of three Q tiles (a pair, then the lone middle, at count 1; the lone middle, then two
    # light rows, at count 2), and odd T over an odd group count (pairs, middle units and a lone middle at count 1, with
    # light rows at count 2).
    CASES = [
        (1, 2, 2, 256, 384, False, False),
        (1, 2, 2, 64, 128, False, False),
        (2, 2, 2, 17, 300, False, False),
        (2, 4, 2, 130, 700, True, True),
        (1, 4, 4, 256, 256, True, False),
        (1, 1, 1, 300, 300, True, False),
        (3, 5, 5, 1152, 1152, True, True),
    ]
    plans = []
    for count in counts:  # 0: the adapter's CTA count; n > 0 forces n persistent CTAs (a compile argument, its own key)
        for b, h_q, h_kv, s_q, s_kv, causal, bottom_right in CASES:
            if switches.get("causal_walk") and not causal:
                continue  # the paired walk orders causal records only
            torch.manual_seed(0)
            q, k, v = (torch.randn(b, s, h, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for s, h in ((s_q, h_q), (s_kv, h_kv), (s_kv, h_kv)))
            o = torch.full((b, s_q, h_q, 128), float("nan"), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
            api = SdpaFwdDslSm90(q, k, v, o, **(dict(is_causal=True, causal_bottom_right=bottom_right) if causal else {}))
            assert api.check_support() and api.flavor == (128, 128), api.flavor
            api.params = dataclasses.replace(api.params, **switches)
            if count:
                api._persistent_ctas = lambda device, n=count: n
            api.compile()
            assert {name: getattr(api._k_mod, name.upper()) for name in switches} == switches
            plans.append((api, q, k, v, o, (count, b, h_q, h_kv, s_q, s_kv, causal, bottom_right)))
    print("COMPILED", len(plans), flush=True)
    faulthandler.dump_traceback_later(run_budget_s, exit=True)
    for api, q, k, v, o, case in plans:
        # Per-launch seconds: a starved launch (slow, then done) reads differently from a wedged one (never done).
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        assert torch.isfinite(o).all(), case
        print("RAN", case, "launch s", " ".join(f"{t:.4f}" for t in times), flush=True)
    faulthandler.cancel_dump_traceback_later()
    print("TURNS OK", flush=True)
    """)
# Run ahead of the child's first import (test/AGENTS.md, timed subprocess tests): a stall while importing or
# compiling dumps the child's stack and exits inside the parent's deadline; the run phase re-arms the dump with its
# own budget. The compile budget is the child's last argument, so the child's own arguments keep their positions.
_TURNS_PREAMBLE = "import faulthandler, sys\nfaulthandler.dump_traceback_later(int(sys.argv[-1]), exit=True)\n"
# The child's whole budget stays inside the conftest's per-test deadline (CUDNN_TEST_TIMEOUT, 0 = none), so a stall
# reaches the HANG report below and the child is killed with it, instead of the worker dying with the child still running.
_TURNS_RUN_BUDGET_S = 120
_TEST_DEADLINE_S = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
_TURNS_COMPILE_BUDGET_S = 900 if _TEST_DEADLINE_S <= 0 else max(60, min(900, int(_TEST_DEADLINE_S) - _TURNS_RUN_BUDGET_S - 60))


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("variant", [pytest.param(v, marks=_level(v in _L0_WATCHDOG)) for v in [*_VARIANTS, *_CAUSAL_VARIANTS]])
def test_d128_turns_and_rings_complete_in_a_watchdogged_child(variant):
    """Every launch of every switch value finishes: no K or V ring slot waits for an arrival that never comes,
    including one KV tile, a warpgroup without rows and an incomplete Q tile, and a persistent record's item loop
    completes at the adapter's CTA count and at one and two CTAs (the paired walk's on the causal cases, through pairs,
    middle units, light rows and a lone middle). A hang is a child that misses its run deadline, reported as HANG with
    its last lines; a child that stalls before every plan has compiled is reported as STALL, with its stack (the
    child arms its own deadline before its first import)."""
    argv = [
        sys.executable,
        "-c",
        _TURNS_PREAMBLE + _TURNS_CHILD,
        json.dumps(_switches(variant)),
        str(_TURNS_RUN_BUDGET_S),
        json.dumps(_counts(variant)),
        str(_TURNS_COMPILE_BUDGET_S),
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=_TURNS_COMPILE_BUDGET_S + _TURNS_RUN_BUDGET_S)
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired carries bytes even with text=True. Keep both streams: a child-side stack dump is on stderr.
        out, err = (s.decode(errors="replace") if isinstance(s, bytes) else (s or "") for s in (exc.stdout, exc.stderr))
        stage = "HANG" if "COMPILED" in out else "STALL before COMPILED"
        budget = _TURNS_COMPILE_BUDGET_S + _TURNS_RUN_BUDGET_S
        pytest.fail(f"{stage} ({variant}): no exit in {budget} s; last lines {out.strip().splitlines()[-3:]}\n--- child stderr ---\n{err[-3000:]}")
    # A child-side dump after COMPILED is a launch that never finished; one before it is an import or compile stall.
    hung = "COMPILED" in proc.stdout and "Timeout" in proc.stderr
    stage = "HANG" if hung else "STALL before COMPILED" if "Timeout" in proc.stderr else "FAILED"
    assert proc.returncode == 0 and "TURNS OK" in proc.stdout, f"{stage} ({variant}), exit {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"


@pytest.mark.L0
def test_d128_adapter_applies_the_shipped_switches(monkeypatch):
    """check_support writes api_dsl's D128 switch constants into a D128 plan's record at call time: persistence only
    under overlap, on a dense plan always and on a causal plan as D128_PERSISTENT_CAUSAL_WALK says (0 one tile per CTA,
    1 the stride by the grid, 2 the walk's rule), the walk's rule's plan going one tile per CTA where
    D128_CAUSAL_WALK_ROUTE routes it, as it routes this two-item plan (at most one item per CTA). A causal record of the
    walk's rule walks the NATURAL order and every other causal record the causal order (_causal_sched_policy), unless
    the caller pins one, which every record keeps. A D512 plan (tile 64/64 pinned) never carries a switch."""
    import itertools

    from cudnn.sdpa.fwd import api_dsl

    q, k, v = _inputs(1, 2, 2, 64, 128, torch.float16)
    o = torch.empty_like(q)
    for overlap, persistent, walk, causal, route in itertools.product((False, True), (False, True), (0, 1, 2), (False, True), (False, True)):
        monkeypatch.setattr(api_dsl, "D128_INTRA_WG_OVERLAP", overlap)
        monkeypatch.setattr(api_dsl, "D128_PERSISTENT", persistent)
        monkeypatch.setattr(api_dsl, "D128_PERSISTENT_CAUSAL_WALK", walk)
        monkeypatch.setattr(api_dsl, "D128_CAUSAL_WALK_ROUTE", route)
        mask = dict(is_causal=True) if causal else {}
        api = api_dsl.SdpaFwdDslSm90(q, k, v, o, **mask)
        assert api.check_support() and api.flavor == (128, 128), api.flavor
        on = overlap and persistent and (not causal or walk > 0)
        paired = on and causal and walk == 2
        got = {name: getattr(api.params, name) for name in _SWITCHES}
        # causal_claim: 0, the two-item plan far below the claim counter's gate (api_dsl.D128_CLAIM_MIN_ROUNDS x CTAs).
        want = dict(intra_wg_overlap=overlap, persistent=on and not (paired and route), causal_walk=int(paired and not route), causal_claim=0)
        assert got == want, (causal, walk, route, got)
        order = SCHED_NATURAL if not causal or (paired and not route) else api_dsl._causal_sched_policy(128, 128, 128, 2)
        assert api.params.sched_policy == order, (causal, walk, route, api.params.sched_policy)
        pinned = api_dsl.SdpaFwdDslSm90(q, k, v, o, sched_policy=SCHED_LPT, **mask)
        assert pinned.check_support() and pinned.params.sched_policy == SCHED_LPT, (causal, walk, route, pinned.params)
        d512 = api_dsl.SdpaFwdDslSm90(q, k, v, o, tile_m=64, tile_n=64, **mask)
        assert d512.check_support() and d512.flavor == (512, 512), d512.flavor
        assert not any(getattr(d512.params, name) for name in _SWITCHES), d512.params


@pytest.mark.L0
def test_d128_adapter_routes_causal_plans_one_tile_per_cta(monkeypatch):
    """check_support with the walk's rule picked (D128_PERSISTENT_CAUSAL_WALK 2), at 132 CTAs (an H100 SXM's SM count,
    patched so that the decision does not move with the board): with D128_CAUSAL_WALK_ROUTE on, a causal plan the route
    sends one tile per CTA (_sm90_d128_causal_route: at most one item per CTA, or L at least _D128_ROUTE_MAX_L) carries
    the record D128_PERSISTENT_CAUSAL_WALK 0 gives it, field for field (the one-tile record, in the causal order), and
    every other plan the record of the walk's rule, in the NATURAL order; with the route off every plan carries the
    latter. L is a plan's mean KV tiles per Q tile. The shapes straddle both of the route's branches (132 and 136 items;
    L from 4.5 to 120.5, MHA, GQA and MQA, both mask alignments); which of them the route sends is its own decision,
    read from the route here and checked against the planner in test_sdpa_fwd_d128_sm90_routing.py.
    """
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl.SdpaFwdDslSm90, "_persistent_ctas", lambda self, device: 132)

    def record(q, k, v, bottom_right, walk, route):
        monkeypatch.setattr(api_dsl, "D128_PERSISTENT_CAUSAL_WALK", walk)
        monkeypatch.setattr(api_dsl, "D128_CAUSAL_WALK_ROUTE", route)
        api = api_dsl.SdpaFwdDslSm90(q, k, v, torch.empty_like(q), is_causal=True, causal_bottom_right=bottom_right)
        assert api.check_support() and api.flavor == (128, 128), api.flavor
        return api.params

    for shape, bottom_right in [
        ((1, 4, 1, 4224, 4224), True),
        ((1, 16, 4, 1024, 1024), True),
        ((1, 4, 4, 4224, 8192), True),
        ((1, 4, 1, 4352, 4352), True),
        ((8, 32, 8, 2048, 16384), True),
        ((8, 32, 8, 1920, 7040), True),
        ((3, 8, 8, 2048, 10240), True),
        ((3, 32, 16, 384, 6272), True),
        ((1, 16, 4, 1280, 6656), True),
        ((8, 24, 8, 256, 2176), True),
        ((13, 16, 16, 1152, 1792), True),
        ((9, 16, 16, 1664, 2688), True),
        ((12, 16, 8, 1280, 2560), True),
        ((8, 16, 4, 1024, 1024), True),
        ((6, 32, 32, 256, 512), True),
        ((3, 8, 8, 768, 4864), True),
        ((8, 32, 8, 2048, 16384), False),
    ]:
        routed = api_dsl._sm90_d128_causal_route(shape, bottom_right, 132)
        q, k, v = _inputs(*shape, torch.bfloat16)
        paired = record(q, k, v, bottom_right, 2, False)
        assert (paired.persistent, paired.causal_walk, paired.sched_policy) == (True, 1, SCHED_NATURAL), (shape, bottom_right, paired)
        assert record(q, k, v, bottom_right, 2, True) == (record(q, k, v, bottom_right, 0, True) if routed else paired), (shape, bottom_right)
        del q, k, v


@pytest.mark.L0
def test_d128_adapter_passes_the_cta_count_exactly_when_persistent(monkeypatch):
    """compile() passes the SM count (SdpaFwdDslSm90._persistent_ctas) to a persistent D128 record and nothing to any
    other plan, so a non-persistent key never names a device."""
    from types import SimpleNamespace

    from cudnn.sdpa.fwd import api_dsl

    class Stop(Exception):
        pass

    calls = []

    def fake_compile(*args, **kw):
        calls.append(kw)
        raise Stop

    monkeypatch.setattr(
        api_dsl, "_load_sm90_kernel_module", lambda flavor, params, native_d64=False: SimpleNamespace(__file__="prefill_d128_f16.py", compile=fake_compile)
    )
    monkeypatch.setattr(api_dsl.SdpaFwdDslSm90, "_persistent_ctas", lambda self, device: 7)
    q, k, v = _inputs(1, 2, 2, 64, 128, torch.float16)
    o = torch.empty_like(q)
    for persistent in (False, True):
        api = api_dsl.SdpaFwdDslSm90(q, k, v, o)
        assert api.check_support() and api.flavor == (128, 128), api.flavor
        api.params = dataclasses.replace(api.params, intra_wg_overlap=True, persistent=persistent)
        with pytest.raises(Stop):
            api.compile()
        kw = calls.pop()
        assert (kw.get("persistent_ctas") == 7) if persistent else ("persistent_ctas" not in kw), kw


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_d128_ships_the_persistent_loop_and_the_causal_route():
    """The shipped record (api_dsl's constants as committed, nothing patched): a dense plan compiles the persistent loop
    over the overlap loop, at the device's SM count; a causal plan the route keeps compiles the record of the walk's
    rule (D128_PERSISTENT_CAUSAL_WALK 2) in the NATURAL order, and one it routes the one-tile-per-CTA overlap kernel
    (D128_CAUSAL_WALK_ROUTE) in the causal order. Each computes the one-tile overlap kernel's O and Stats bit for bit.
    The causal plans take the same decision on any SM90 GPU of 32 to 132 SMs: b1 h4/1 S 1024 has 32 items (8 Q tiles of
    4 heads), so the route sends it one tile per CTA; b4 h16/4 S 512 (256 items, T 4: the NATURAL stride) and b2 h16/4 S
    1152 (288 items, T 9: the paired walk) have more items than an SM90 GPU has SMs, so the route keeps them."""
    from cudnn.sdpa.fwd import api_dsl

    for shape, mask, want in [
        ((2, 4, 2, 300, 700), {}, (True, True, 0)),
        ((1, 4, 1, 1024, 1024), dict(is_causal=True), (True, False, 0)),
        ((4, 16, 4, 512, 512), dict(is_causal=True), (True, True, 1)),
        ((2, 16, 4, 1152, 1152), dict(is_causal=True), (True, True, 1)),
    ]:
        q, k, v = _inputs(*shape, torch.bfloat16)
        api, o, lse = _plan(q, k, v, stats=True, **mask)
        assert (api._k_mod.INTRA_WG_OVERLAP, api._k_mod.PERSISTENT, api._k_mod.CAUSAL_WALK) == want, (shape, mask)
        order = api_dsl._causal_sched_policy(shape[4], 128, 128, 2) if mask and not want[1] else SCHED_NATURAL
        assert api.params.sched_policy == order, (shape, mask, api.params.sched_policy)
        if want[2]:
            # the walk's rule at the device's SM count: the stride below CAUSAL_WALK_MIN_TILES Q tiles, the walk at T 9
            # (at every count from 32 to 132)
            assert _walk_branch(api, None).startswith("stride") is (shape[3] < 1152), (shape, _walk_branch(api, None))
        if want[1]:
            assert api._persistent_ctas(q.device) == torch.cuda.get_device_properties(q.device).multi_processor_count
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
        torch.cuda.synchronize()
        base = _run(q, k, v, stats=True, variant="ov", **mask)
        _assert_bitwise("the shipped record", f"{'causal' if mask else 'dense'} {shape} {mask}", (o, lse), base, "ov")


# --- Tight numerics against FP64 -------


def _ref64(q, k, v, *, scale, is_causal, bottom_right):
    """FP64 O and natural-log Stats (no TF32 anywhere: both matmuls run in double)."""
    b, h_q, s_q, _ = q.shape
    group = h_q // k.shape[1]
    kk, vv = (t.repeat_interleave(group, dim=1).double() for t in (k, v))
    scores = torch.matmul(q.double(), kk.transpose(-1, -2)) * scale
    if is_causal:
        s_kv = k.shape[2]
        rows = torch.arange(s_q, device=q.device).view(-1, 1) + (s_kv - s_q if bottom_right else 0)
        scores = scores.masked_fill(torch.arange(s_kv, device=q.device).view(1, -1) > rows, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vv), torch.logsumexp(scores, dim=-1)


@pytest.mark.L0
@pytest.mark.parametrize("shape", [(2, 8, 2, 256, 256), (1, 8, 8, 2000, 1000)], ids=["b2-gqa-s256", "b1-mha-sq2000-skv1000"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_d128_fp64_bounds(shape, causal, dtype):
    """|O - O64| <= u * (|O64| + max|V|) with a 2x margin, u the input dtype's unit roundoff (P
    is rounded to it for the 16-bit WGMMA and O is stored in it); Stats within 1e-3 + 1e-5 rel."""
    b, h_q, h_kv, s_q, s_kv = shape
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    o, lse = _run(q, k, v, stats=True, is_causal=causal)
    o64, lse64 = _ref64(q, k, v, scale=_SCALE, is_causal=causal, bottom_right=s_q == s_kv)
    u = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-11
    bound = 2 * u * (o64.abs() + v.double().abs().max())
    excess = (o.double() - o64).abs() - bound
    assert excess.max().item() <= 0, f"O exceeds its FP64 bound by up to {excess.max().item():.3e}"
    torch.testing.assert_close(lse.double(), lse64, atol=1e-3, rtol=1e-5)


# --- Large scores: the FP32 exponent rounding -------

# Q and K hold one bf16 constant a = 1.5 * 2**k, so every score is M = 128 * a * a: exact in bf16 products and in every
# FP32 partial sum (at most 11 significant bits), whatever order the WGMMA adds them in.
_LARGE_SCORE_AMPS = [1.5 * 2.0**k for k in (0, 6, 12, 14, 16, 17)]


def _row_max_id(a):
    mantissa, exponent = f"{128 * a * a:.1e}".split("e")
    return f"max-{mantissa}e{int(exponent)}"


@pytest.mark.L0
@pytest.mark.parametrize("a", _LARGE_SCORE_AMPS, ids=[_row_max_id(a) for a in _LARGE_SCORE_AMPS])
@torch_fork_set_rng(seed=0)
def test_d128_large_scores_keep_rows_finite_and_nonzero(a):
    """Every score equals M = 128 * a * a, so the softmax is uniform: O is the mean of V over the 256 keys and Stats is
    M * scale + ln(256). The FP32 path forms each exponent as one FFMA, s * k - c with k = scale * log2(e) and c = M * k
    rounded once per row, so the top score's exponent is c's rounding error instead of 0: about -16 at M = 4.8e9, -256 or
    less from M = 7.7e10 on, where every P underflows and normalize_row takes its empty-row branch (O = 0, Stats = -inf).
    With the row limit at 2**24 / |k| (about 1.3e8 here) every case from M = 4.8e9 on takes the FP64 exponents instead;
    with only the old overflow limit (2**126) the three largest cases fail. This test is the detector."""
    q, k, v = _inputs(1, 2, 2, 128, 256, torch.bfloat16)
    q.fill_(a)
    k.fill_(a)
    o, lse = _run(q, k, v, stats=True)
    o64, lse64 = _ref64(q, k, v, scale=_SCALE, is_causal=False, bottom_right=False)
    bad_o = (~torch.isfinite(o)).any(dim=-1)
    zero_o = (o == 0).all(dim=-1)
    bad_lse = ~torch.isfinite(lse)
    assert not (bad_o.any() or zero_o.any() or bad_lse.any()), (
        f"row max {128 * a * a:.3e}: {int(bad_o.sum())} rows with a non-finite O, {int(zero_o.sum())} all-zero O rows "
        f"(V is random, the exact O is its mean), {int(bad_lse.sum())} non-finite Stats of {lse.numel()} rows"
    )
    u = 2.0**-8
    excess = (o.double() - o64).abs() - 2 * u * (o64.abs() + v.double().abs().max())
    assert excess.max().item() <= 0, f"row max {128 * a * a:.3e}: O exceeds its FP64 bound by up to {excess.max().item():.3e}"
    torch.testing.assert_close(lse.double(), lse64, atol=0.0, rtol=1e-5)


# --- Code generation: the register split, spills and WGMMA serialization -------

_SASS_PROBE = textwrap.dedent(r"""
    import dataclasses, glob, json, os, re, subprocess, sys
    dump, case, ptxas, cands = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_ARCH"] = "sm_90a"  # unconditional: an inherited value would pin another target
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template
    from cudnn.sdpa.fwd.config_sm90 import TemplateParams
    # The heaviest D128 body: GQA and Stats, causal (three masked phases) or dense (a persistent record's dense body is
    # different code); ``fields`` is the variant's switch record, ``ctas`` a persistent record's CTA count, ``shape``
    # the declared (b, h_q, h_kv, s_q, s_kv), whose head counts give the GQA ratio, and ``sched`` an order other than
    # TemplateParams' NATURAL, where the case names one.
    fields = case["switches"]
    b, h_q, h_kv, s_q, s_kv = case["shape"]
    order = {"sched_policy": case["sched"]} if "sched" in case else {}
    params = dataclasses.replace(TemplateParams(dtype_qkv=DTYPE_BF16, causal=case["causal"], has_lse=True, qh_per_kh=h_q // h_kv, **order), **fields)
    mod = _load_kernel_template("sm90/prefill_d128_f16.py", params, tag="sass_sm90_d128")
    print("SWITCHES", *(int(getattr(mod, name.upper())) for name in sorted(fields)))
    if fields.get("causal_walk"):
        # The walk the record takes: the template's rule over the probe's record, shape and CTA count, as the kernel's
        # constructor calls it (bf16: 2 bytes). Light rows, pairs, middles and the lone middle; -1 for the stride.
        bottom_right = params.causal if params.bottom_right is None else params.bottom_right
        walk = mod.causal_walk_plan((b, h_q, h_kv, s_q, s_kv), bottom_right, params.sched_policy, params.qh_per_kh, case["ctas"], 2)
        print("WALK", *((-1,) * 4 if walk is None else (walk.light_rows, walk.pairs, int(walk.middles), walk.lone_middle)))
        if walk is None:
            # The stride's deal the record takes (causal_stride_snake): 1 deals odd rounds in reverse.
            deal = mod.causal_stride_snake((b, h_q, h_kv, s_q, s_kv), bottom_right, params.sched_policy, params.qh_per_kh, case["ctas"], 2)
            print("DEAL", int(deal))
    q, kv = (s_q * h_q * 128, 128, h_q * 128, 1), (s_kv * h_kv * 128, 128, h_kv * 128, 1)
    mod.compile(b, h_q, h_kv, s_q, s_kv, q, kv, kv, q, (h_q * s_q, s_q, 1), **({"persistent_ctas": case["ctas"]} if fields["persistent"] else {}))
    cubins = sorted(glob.glob(os.path.join(dump, "**", "*.cubin"), recursive=True), key=os.path.getmtime)
    ptxs = sorted(glob.glob(os.path.join(dump, "**", "*.ptx"), recursive=True), key=os.path.getmtime)
    if not cubins or not ptxs:
        print("FAIL no cubin/ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    text = [ln for ln in sass if re.match(r"\s+/\*[0-9a-f]{4,}\*/", ln)]
    keys = {"USETMAXREG": "USETMAXREG", "STL": "STL", "LDL": "LDL", "HGMMA": "HGMMA", "DEPBAR": "WARPGROUP.DEPBAR",
            "DEPBAR_1": "DEPBAR.LE gsb0, 0x1", "BAR_SYNC": "BAR.SYNC", "BAR_ARV": "BAR.ARV"}
    for name, key in keys.items():
        print("SASS", name, sum(1 for ln in sass if key in ln))
    print("SASS MAXREG", max((int(r) for ln in text for r in re.findall(r"\bR(\d+)\b", ln)), default=-1))
    # Overlap's reach: MUFU.EX2 between a wait_group(1) and the next wait_group(0), the fewest over the steps (34 of
    # 66 in an offline compile of the ov build). Printed to compare builds, not pinned.
    under = []
    for a, ln in enumerate(text):
        if "DEPBAR.LE gsb0, 0x1" in ln:
            z = next((j for j in range(a + 1, len(text)) if "DEPBAR.LE gsb0, 0x0" in text[j]), len(text))
            under.append(sum("MUFU.EX2" in t for t in text[a:z]))
    print("SASS EX2_UNDER_PV", min(under) if under else -1)
    alloc = [int(ln.split(",")[-1].strip(" ;"), 16) for ln in sass if "USETMAXREG.TRY_ALLOC" in ln]
    dealloc = [int(ln.split()[-2], 16) for ln in sass if "USETMAXREG.DEALLOC" in ln]
    print("SASS USETMAXREG_ALLOC", alloc[0] if alloc else -1)
    print("SASS USETMAXREG_DEALLOC", dealloc[0] if dealloc else -1)
    # FP64 instructions on the narrow path of the compute KV loop, which softmax_setup and online_softmax keep free of
    # them: there an FP64 instruction waits behind the other warpgroup's WGMMA on every step. A loop is a backward
    # branch; a KV loop holds an HGMMA, no EXIT (the try-wait retry stubs past the kernel's end also branch back) and no
    # O store (UTMASTG or STG: past a fully peeled KV loop the item loop would be the innermost), and no smaller such
    # loop. Its narrow path takes each predicated branch on a predicate operand forward (the wide test, so the wide arms
    # and their FP64 exponents are left out) over the whole loop: the head that issues QK and PV, and both row halves.
    # Counted: DMUL, DADD, DFMA, DSETP, DMNMX, an F2F to or from F64, MUFU.RCP64H and MUFU.RSQ64H.
    labels, ins, pend = {}, [], []
    for ln in sass:
        m = re.match(r"\s*(\.L_x_\d+):", ln)
        if m:
            pend.append(m.group(1))
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*$", ln)
        if m:
            addr, body = int(m.group(1), 16), m.group(2)
            labels.update((p, addr) for p in pend)
            pend = []
            pred, rest = body.split(None, 1) if body.startswith("@") else ("", body)
            ins.append((addr, pred, rest.split()[0] if rest else "", body))
    spans = []
    for addr, pred, op, body in ins:
        m = re.search(r"\(\s*(\.L_x_\d+)\s*\)", body)
        if op == "BRA" and m and labels.get(m.group(1), addr + 1) <= addr:
            span = [i for i in ins if labels[m.group(1)] <= i[0] <= addr]
            if any(i[2].startswith("HGMMA") for i in span) and not any(i[2] == "EXIT" or i[2].startswith(("UTMASTG", "STG")) for i in span):
                spans.append((labels[m.group(1)], addr))
    loops = [s for s in spans if not any(t != s and s[0] <= t[0] and t[1] <= s[1] for t in spans)]
    fp64 = []
    for lo, hi in loops:
        path, skip = [], None
        for addr, pred, op, body in ins:
            if not lo <= addr <= hi or (skip is not None and addr < skip):
                continue
            skip = None
            path.append((addr, op, body))
            m = re.search(r"BRA (!?P\d), `\((\.L_x_\d+)\)", body)
            if op == "BRA" and pred and m and labels.get(m.group(2), -1) > addr:
                skip = labels[m.group(2)]
        for addr, op, body in path:
            if re.match(r"(DMUL|DADD|DFMA|DSETP|DMNMX|MUFU\.RCP64H|MUFU\.RSQ64H)\b", op) or (op.startswith("F2F") and "F64" in op):
                fp64.append(f"{op}@{addr:#x}")
    print("SASS KV_LOOPS", len(loops))
    print("SASS FP64_KV_LOOP", len(fp64))
    print("FP64", *fp64)
    ptx = open(ptxs[-1], encoding="utf-8", errors="replace").read()
    pred = r"^\s*(?:@!?%p\d+\s+)?"
    aligned = re.findall(pred + r"(?:bar(?:\.cta)?\.(?:sync|arrive)\b|barrier(?:\.cta)?\.(?:sync|arrive)\.aligned\b)", ptx, re.M)
    print("PTX LEGACY_BAR", len(aligned))
    print("PTX BARRIER_ARRIVE", len(re.findall(pred + r"barrier(?:\.cta)?\.arrive\b", ptx, re.M)))
    if ptxas == "-":
        print("PTXAS none"); sys.exit(0)
    asm = subprocess.run([ptxas, "-arch=sm_90a", "-v", "-o", os.path.join(dump, "probe.cubin"), ptxs[-1]], capture_output=True, text=True, timeout=900)
    log = asm.stdout + asm.stderr
    if asm.returncode != 0 and "Unsupported .version" in log:
        print("PTXAS too_old"); sys.exit(0)
    print("PTXAS rc", asm.returncode)
    print("PTXAS serialized", sum(1 for ln in log.splitlines() if "wgmma.mma_async instructions are serialized" in ln))
    used = re.search(r"used (\d+) barriers", log)
    print("PTXAS barriers", used.group(1) if used else -1)
    for ln in log.splitlines():
        if "Used" in ln or "spill" in ln or "C75" in ln:
            print("PTXAS_LINE", ln.strip())
    """)
# No spills: each case below compiles to 0 STL and 0 LDL at the 240-register budget in an offline sm_90a compile (ptxas
# 13.4 V13.4.92, CuTe DSL 4.8.0); the test holds every new compile to that, within SPILL_TOLERANCE.
_SPILL_PIN = {"STL": 0, "LDL": 0}
# The SASS cases: (variant, causal body). Today's two on the causal body, plus the persistent record on the dense body,
# which the adapter ships on non-causal plans (one step instance: the masked phases fold away), and the paired walk,
# which the walk's rule runs on causal plans of at least CAUSAL_WALK_MIN_TILES Q tiles the route keeps (ov is the record
# of those it routes), on the causal body (b1 h8/2 S 2048 at 64 CTAs, T 16: pairs alone, eight pairs per group, one unit
# per CTA); the stride on the one-Q-tile body (S_q 100, S_kv 300, NATURAL, 3 CTAs), whose Q tile work_of writes as a
# division so that the item's loop bounds stay run-time values (as a trace-time constant it spilled; work_of's
# comment). The last two are on a chunked prefill body (bottom-right, S_kv 1792 past S_q, LPT_L2, an order a caller may
# pin): the stride by the grid at 132 CTAs (b8 h24/8 S_q 256 S_kv 2048) and the paired walk at 3 CTAs (b8 h24/8 S_q
# 1280 S_kv 3072, T 10: pairs alone, five per group), the records on which NVVM had recomputed the FP64 log2 scale
# inside the KV loop (softmax_setup's comment). The paired walk's one-Q-tile case (middle tiles only) is gone: the T
# cut keeps a T 1 plan on the stride, which compiles pa-causal-t1's code. The claim counter's record
# (causal_claim 1) on the probe's causal body at 64 CTAs (128 items, two per CTA), held to the rules every case is
# held to; its FP64 guard, loop and body counts are measured on the first H100 that compiles it (not pinned yet).
_SASS_CASES = {
    "v1": ("v1", True),
    "ov": ("ov", True),
    "pa-dense": ("pa", False),
    "pw-causal": ("pw", True),
    "pa-causal-t1": ("pa", True),
    "pa-causal-t2": ("pa", True),
    "pw-causal-chunked": ("pw", True),
    "pn-causal-snake": ("pw", True),
    "pn-causal-straight": ("pw", True),
    "pc1-causal": ("pc1", True),
}
# The NATURAL stride a causal record of the walk's rule keeps below CAUSAL_WALK_MIN_TILES, on the probe's causal body
# (T 8, NATURAL, TemplateParams' order) under each deal: at 64 CTAs a CTA's two items take one Q tile position under the
# straight deal, so the rule deals odd rounds in reverse (pn-causal-snake); at 127 CTAs the straight deal is no heavier
# (pn-causal-straight). Both are held to the rules every case is held to (the register split, 0 spills, the
# 240-register budget, wait_group(1) under overlap), to no FP64 instruction on the narrow path of their compute KV
# loops, and to the loop and body counts an H100 compile measured (_KV_LOOPS, _CODEGEN_PINS).
_SASS_DEALS = {"pn-causal-snake": 1, "pn-causal-straight": 0}
# A persistent record's CTA count in the probe: below the probe's 128 items, so each CTA walks two. With the count at or
# above the items every CTA runs one item, and LLVM could fold the item loop away if it ever proved that (the warp-uniform
# item index keeps it from doing so today); a count below the items pins the loop whatever it proves.
_SASS_CTAS = 64
# The probe's declared problem (b, h_q, h_kv, s_q, s_kv), and the cases that compile another one at their own CTA
# count.
_SASS_SHAPE = (2, 8, 2, 1024, 1024)
_SASS_SHAPES = {
    "pw-causal": ((1, 8, 2, 2048, 2048), 64),
    "pa-causal-t1": ((2, 8, 2, 100, 300), 3),
    "pa-causal-t2": ((8, 24, 8, 256, 2048), 132),
    "pw-causal-chunked": ((8, 24, 8, 1280, 3072), 3),
    "pn-causal-straight": (_SASS_SHAPE, 127),
}
# The cases compiled in another order than TemplateParams' NATURAL: the chunked prefill ones in LPT_L2.
_SASS_ORDERS = {"pa-causal-t2": SCHED_LPT_L2, "pw-causal-chunked": SCHED_LPT_L2}
# The walk each paired-walk case compiles (the probe's WALK line: light rows, pairs, middles, lone middle), so that a
# rule or constant change that moves a case to another branch, or to the stride (-1), fails here instead of pinning
# another body.
_SASS_WALKS = {"pw-causal": (0, 8, 0, 0), "pw-causal-chunked": (0, 5, 0, 0), **{case: (-1,) * 4 for case in _SASS_DEALS}}
# The causal persistent cases, held to no FP64 instruction on the narrow path of the compute KV loop (the probe's
# FP64_KV_LOOP), with the KV loops their bodies keep (KV_LOOPS), so that the count reads the loops it is about: two on
# pw-causal, three on the one-Q-tile stride, one on the chunked prefill stride (T 2), two on the chunked prefill walk
# (T 10) and two on each NATURAL stride (T 8). The walk's T 10 and the NATURAL strides' counts are from a compile on
# an H100 SXM (2026-11-06, CuTe DSL 4.8.0 with its cu13 libraries: FP64 0, LDL and STL 0, R234 on pw-causal-chunked and
# R236 on both NATURAL strides); the others the offline compile's of _CODEGEN_PINS, which that compile matched. pw-causal-chunked had
# kept one loop at T 2 (S_q 256 S_kv 2048, its shape before the T cut moved that plan to the stride). In that offline
# compile of the kernel before softmax_setup pinned the scale, both chunked prefill cases held a conversion and a
# multiply on that path (F2F.F64.F32 and DMUL), pw-causal and pa-causal-t1 none.
_FP64_GUARD = ("pw-causal", "pa-causal-t1", "pa-causal-t2", "pw-causal-chunked", *_SASS_DEALS)
_KV_LOOPS = {"pw-causal": 2, "pa-causal-t1": 3, "pa-causal-t2": 1, "pw-causal-chunked": 2, **{case: 2 for case in _SASS_DEALS}}
# Per-case structure of the probe body (GQA 4, Stats, bf16), measured in an offline sm_90a compile (ptxas 13.4 V13.4.92,
# CuTe DSL 4.8.0): v1 = 3 step instances x (8 SS + 8 RS); ov = prologue QK 8 + 2 step instances x 16 + drain PV 8, one
# wait_group(1) per step instance; pa-dense is one copy of ov's dense item body inside the item loop, one step instance
# (the probe body made non-causal, persistent at 64 CTAs). BAR_SYNC: the CTA barrier after the barrier init and the O
# staging barrier in each branch of the warpgroup split (3); pa-dense adds, in each branch, the barrier that holds warps
# 1 to 3 until warp 0's deferred O store wait returns (5). pw-causal is one copy of ov's causal item body inside the
# item loop (48, 6, 2) with pa-dense's five barriers; on S 1024 it peaked at R232, where the stride peaks at R236 on
# that body. pa-causal-t1, the stride over the one-Q-tile body, whose partial Q tile and ragged KV trace one more step
# instance (64, 8, 3; ov's one-tile kernel on that shape has the same three; R234). pa-causal-t2 and pw-causal-chunked,
# the chunked prefill body (GQA 3, LPT_L2) under the stride and the paired walk, have pw-causal's counts. pw-causal and
# pw-causal-chunked were measured on S 1024 and S_q 256 S_kv 2048 (T 8 and 2, pairs alone at 64 and 3 CTAs, before the T
# cut moved those plans to the stride): the walk's branch flags are the same at T 16 and 10, so the trace is the same
# code with other constants, and their counts carry over until the next offline compile re-measures them (the H100
# compile found the same counts at T 16 and 10). The NATURAL strides (_SASS_DEALS) have pa-causal-t2's counts there:
# the causal item body inside the item loop. Body counts are a toolchain fact (python/cudnn/sdpa/AGENTS.md, the LDTM
# lesson): re-measure on a new toolchain and record it next to the numbers.
_CODEGEN_PINS = {
    "v1": dict(HGMMA=48, DEPBAR=6, DEPBAR_1=0, BAR_SYNC=3, BAR_ARV=0),
    "ov": dict(HGMMA=48, DEPBAR=6, DEPBAR_1=2, BAR_SYNC=3, BAR_ARV=0),
    "pa-dense": dict(HGMMA=32, DEPBAR=4, DEPBAR_1=1, BAR_SYNC=5, BAR_ARV=0),
    "pw-causal": dict(HGMMA=48, DEPBAR=6, DEPBAR_1=2, BAR_SYNC=5, BAR_ARV=0),
    "pa-causal-t1": dict(HGMMA=64, DEPBAR=8, DEPBAR_1=3, BAR_SYNC=5, BAR_ARV=0),
    "pa-causal-t2": dict(HGMMA=48, DEPBAR=6, DEPBAR_1=2, BAR_SYNC=5, BAR_ARV=0),
    "pw-causal-chunked": dict(HGMMA=48, DEPBAR=6, DEPBAR_1=2, BAR_SYNC=5, BAR_ARV=0),
    **{case: dict(HGMMA=48, DEPBAR=6, DEPBAR_1=2, BAR_SYNC=5, BAR_ARV=0) for case in _SASS_DEALS},
}


def _ptxas():
    import os
    import shutil

    cands = [os.path.join(os.environ["CUDA_PATH"], "bin", "ptxas")] if os.environ.get("CUDA_PATH") else []
    cands.append(shutil.which("ptxas") or "")
    return next((c for c in cands if c and os.path.isfile(c) and os.access(c, os.X_OK)), None)


def _assert_claim_release_rendering(case, switches, dump):
    """Rule 11's rendering pin on the probe's PTX (the kernel's, the newest file in ``dump``, as the probe reads it).
    The id ring's EMPTY arrive is the template's one release of a stage that ordinary loads read; tile_dsl.consume's
    elected form spells it ``@%p mbarrier.arrive.shared::cta.b64 _, [addr]`` (the native arrive writes a state
    register), and the optimizer may copy it with the walk's loop, so every copy is counted and each must be computed
    from at least one loaded shared-memory register through an ``and`` with the kernel's LAST parameter
    (release_zero_arg)."""
    import warm_stress

    ptxs = sorted(dump.rglob("*.ptx"), key=os.path.getmtime)
    assert ptxs, f"{case}: no PTX dumped into {dump}"
    ptx = ptxs[-1].read_text(encoding="utf-8", errors="replace")
    sites = warm_stress.release_dependence(ptx)
    helper = [site for site in sites if re.search(r"mbarrier\.arrive\.shared::cta\.b64\s+_\s*,", site.text)]
    listing = "\n".join(f"  line {s.line}: dependent={s.dependent} loaded registers={s.loaded_registers} param={s.parameter} :: {s.text}" for s in sites)
    if not switches["causal_claim"]:
        assert not helper and not any(site.dependent for site in sites), f"{case}: a release-zero arrive on a record without the claim ring:\n{listing}"
        return
    assert helper, f"{case}: no tile_dsl.consume arrive in the claim record's PTX:\n{listing}"
    assert all(site.text.startswith("@%p") for site in helper), f"{case}: the id ring's EMPTY arrive must be the elected (predicated) form:\n{listing}"
    warm_stress.assert_release_dependence(ptx, sites=len(helper), words=1)
    entry = re.search(r"\.entry\s+(\w+)", ptx)
    indices = [int(i) for i in re.findall(rf"\b{re.escape(entry.group(1))}_param_(\d+)\b", ptx)] if entry else []
    last = f"{entry.group(1)}_param_{max(indices)}" if indices else None
    assert last is not None and all(site.parameter == last for site in helper), f"{case}: the mask is not the kernel's last parameter ({last}):\n{listing}"


@pytest.mark.parametrize("case", [pytest.param(c, marks=_level(c in _L0_SASS)) for c in _SASS_CASES])
def test_d128_register_split_no_spills_and_no_wgmma_serialization(tmp_path, case):
    """The (24, 240, 240) setmaxnreg split survives into SASS (min_blocks_per_mp=1 on the launch;
    python/cudnn/AGENTS.md: ptxas otherwise drops it silently), the compute body does not spill and stays
    under the 240-register budget, and ptxas keeps the QK and PV WGMMA groups pipelined (no C7511/C7513
    serialization advisory), for every switch record and body in _SASS_CASES. The switch must really
    compile: a wait_group(1) appears exactly under overlap. A causal persistent case (_FP64_GUARD) holds no FP64
    instruction on the narrow path of its compute KV loop, in the loops _KV_LOOPS counts. The rendering pin of
    python/cudnn/AGENTS.md Rule 11 (python/cudnn/sdpa/AGENTS.md S15) reads the same PTX: the claim record's id-ring
    EMPTY arrives (tile_dsl.consume, the predicated ``mbarrier.arrive.shared::cta.b64 _, [...]``) are each computed
    from the loaded id through an ``and`` with the kernel's last parameter, the release-zero
    (warm_stress.assert_release_dependence); every other record has no such arrive and no dependent site."""
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / "dump"
    dump.mkdir()
    variant, causal = _SASS_CASES[case]
    switches = _switches(variant)
    overlap = switches["intra_wg_overlap"]
    shape, ctas = _SASS_SHAPES.get(case, (_SASS_SHAPE, _SASS_CTAS))
    order = {"sched": _SASS_ORDERS[case]} if case in _SASS_ORDERS else {}
    arg = json.dumps({"switches": switches, "causal": causal, "ctas": ctas, "shape": shape, **order})
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, str(dump), arg, _ptxas() or "-", *cands], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"sm_90a trace-compile of the D128 template ({case}) failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    _assert_claim_release_rendering(case, switches, dump)
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    want = "SWITCHES " + " ".join(str(int(switches[name])) for name in sorted(switches))
    assert want in out, [ln for ln in out if ln.startswith("SWITCHES")]
    if switches["causal_walk"]:
        walk = "WALK " + " ".join(map(str, _SASS_WALKS[case]))
        assert walk in out, (case, walk, [ln for ln in out if ln.startswith("WALK")])
        if case in _SASS_DEALS:
            assert f"DEAL {_SASS_DEALS[case]}" in out, (case, [ln for ln in out if ln.startswith("DEAL")])
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith(("SASS ", "PTX "))}
    print(f"\nD128 SASS {case}:", stats, [ln for ln in out if ln.startswith("PTXAS")])
    assert stats["USETMAXREG"] > 0, "no USETMAXREG: ptxas dropped the register split (C7508)"
    assert (stats["USETMAXREG_ALLOC"], stats["USETMAXREG_DEALLOC"]) == (240, 24), stats
    assert stats["HGMMA"] > 0, stats
    # A named barrier reached from a warp-role branch is a subset barrier and lowers without the aligned-CTA promise
    # (python/cudnn/AGENTS.md): the O staging barriers of each compute warpgroup go through warpgroup_barrier_sync, so
    # the only aligned barrier left in the PTX is the whole-CTA publish after the barrier init.
    assert stats["LEGACY_BAR"] <= 1, f"{case}: {stats['LEGACY_BAR']} aligned bar.sync / barrier.sync.aligned in the PTX"
    assert_no_new_spills(stats, _SPILL_PIN, tag=f"D128 sm_90a {case}: ")
    assert stats["MAXREG"] < 240, f"{case}: register R{stats['MAXREG']} is past the 240-register budget"
    assert (stats["DEPBAR_1"] > 0) == overlap, f"{case}: wait_group(1) count {stats['DEPBAR_1']}"
    if case in _CODEGEN_PINS:
        assert {key: stats[key] for key in _CODEGEN_PINS[case]} == _CODEGEN_PINS[case], (case, stats)
    if case in _FP64_GUARD:
        if case in _KV_LOOPS:
            assert stats["KV_LOOPS"] == _KV_LOOPS[case], f"{case}: {stats['KV_LOOPS']} compute KV loops, not {_KV_LOOPS[case]}: the FP64 count reads other code"
        fp64 = next((ln.split()[1:] for ln in out if ln.split()[:1] == ["FP64"]), None)
        assert stats["FP64_KV_LOOP"] == 0, (
            f"{case}: FP64 instructions on the narrow path of the compute KV loop, where each waits behind the other warpgroup's WGMMA on every step "
            f"(softmax_setup's comment): {fp64}"
        )
    ptxas = {ln.split()[1]: ln.split()[2] for ln in out if ln.startswith("PTXAS ") and len(ln.split()) == 3}
    if "serialized" in ptxas:
        assert ptxas["rc"] == "0" and ptxas["serialized"] == "0", [ln for ln in out if ln.startswith("PTXAS")]
