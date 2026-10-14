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


# TemplateParams.intra_wg_overlap per step 7 switch value.
_VARIANTS = {"v1": False, "ov": True}


def _plan(q, k, v, *, stats=False, variant=None, **kw):
    """One compiled D128 plan over NaN-filled O (and Stats); asserts the adapter chose D128.

    ``variant`` names a step 7 switch value (``_VARIANTS``), set on the plan's record between
    ``check_support`` and ``compile``; the compiled module must carry it, so a cache-key slip that hands
    back the other value's kernel fails here. None is the adapter's shipped value.
    """
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h_q, s_q, d = q.shape
    o = torch.full((b, s_q, h_q, d), float("nan"), dtype=q.dtype, device="cuda").transpose(1, 2)
    lse = torch.full((b, h_q, s_q), float("nan"), dtype=torch.float32, device="cuda") if stats else None
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, **kw)
    assert api.check_support() and api.flavor == (128, 128), api.flavor
    if variant is not None:
        api.params = dataclasses.replace(api.params, intra_wg_overlap=_VARIANTS[variant])
    api.compile()
    if variant is not None:
        assert api._k_mod.INTRA_WG_OVERLAP == _VARIANTS[variant], variant
    return api, o, lse


def _run(q, k, v, *, stats=False, variant=None, **kw):
    """Plan, execute once, and return (O, Stats or None)."""
    api, o, lse = _plan(q, k, v, stats=stats, variant=variant, **kw)
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


@torch_fork_set_rng(seed=0)
def test_d128_schedulers_are_bitwise_equal():
    """NATURAL, LPT and LPT_L2 launch the same tiles in different orders over ~4 waves on 132 SMs;
    each tile's arithmetic is identical, so outputs are equal bit for bit."""
    q, k, v = _inputs(4, 16, 4, 1024, 1024, torch.bfloat16)
    runs = [_run(q, k, v, stats=True, is_causal=True, sched_policy=policy) for policy in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2)]
    for o, lse in runs[1:]:
        assert torch.equal(o, runs[0][0]) and torch.equal(lse, runs[0][1])
    _check(*runs[0], q, k, v, is_causal=True)


@pytest.mark.parametrize("variant", list(_VARIANTS))
@torch_fork_set_rng(seed=0)
def test_d128_repeat_launches_are_bitwise_equal(variant):
    """Two launches of one plan over the same inputs agree bit for bit (no race, no stale stage), for every
    step 7 switch value."""
    q, k, v = _inputs(2, 8, 2, 700, 900, torch.float16)
    api, o, lse = _plan(q, k, v, stats=True, is_causal=True, variant=variant)
    outs = []
    for _ in range(2):
        o.fill_(float("nan"))
        lse.fill_(float("nan"))
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse)
        torch.cuda.synchronize()
        outs.append((o.clone(), lse.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


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


# --- Step 7: every switch value computes one function -------
#
# Overlap reorders issues and waits around the same multiplies, P words, V stages and k-step chains (plan
# step7-overlap-pingpong.md 3.2), so it equals v1 bit for bit. Each case is an edge of the overlap protocol.

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
}


def _reference_mask(kw, s_q, s_kv):
    """(is_causal, bottom_right) for _check: the adapter spells top-left on a dense square as bottom-right."""
    causal = kw.get("is_causal", False)
    return causal, causal and (kw.get("causal_bottom_right", False) or s_q == s_kv)


@pytest.mark.parametrize("case", list(_CROSS))
@pytest.mark.parametrize("variant", ["ov"])
@torch_fork_set_rng(seed=0)
def test_d128_variant_is_bitwise_v1(variant, case):
    """Every step 7 switch value computes v1's O and Stats bit for bit; v1 itself is checked against the
    reference. A difference that repeats launch after launch is arithmetic (compare the softmax block's
    FFMA/FMUL/FADD counts in SASS); one that moves between launches is a race."""
    b, h_q, h_kv, s_q, s_kv, dtype, kw = _CROSS[case]
    q, k, v = _inputs(b, h_q, h_kv, s_q, s_kv, dtype)
    if case == "wide-fp64":
        # A row's four lanes share its path, and a warp holds rows of both parities, so its lanes split.
        q[:, :, 0::2].fill_(_WIDE)
        k.fill_(_WIDE)
    base = _run(q, k, v, stats=True, variant="v1", **kw)
    if case != "wide-fp64":  # its FP64 check is test_d128_large_scores_keep_rows_finite_and_nonzero's
        is_causal, bottom_right = _reference_mask(kw, s_q, s_kv)
        _check(*base, q, k, v, scale=kw.get("scale_softmax", _SCALE), is_causal=is_causal, bottom_right=bottom_right)
    o, lse = _run(q, k, v, stats=True, variant=variant, **kw)
    if not (torch.equal(o, base[0]) and torch.equal(lse, base[1])):
        d_o = (o.float() - base[0].float()).abs()
        rows = torch.nonzero(d_o.amax(dim=-1) > 0)[:8].tolist()
        pytest.fail(
            f"{variant} is not v1 on {case}: max |dO| {d_o.max().item():.3e}, Stats equal {torch.equal(lse, base[1])}, "
            f"first differing (batch, head, row) {rows}; rerun test_d128_repeat_launches_are_bitwise_equal[{variant}]: "
            "stable launches mean arithmetic (compare the softmax FFMA/FMUL/FADD counts), moving ones a race"
        )


@pytest.mark.parametrize("variant", list(_VARIANTS))
@torch_fork_set_rng(seed=0)
def test_d128_graph_replay_is_bitwise_eager(variant):
    """CUDA-graph replay of one plan equals its eager launch bit for bit, before and after the inputs change
    in place at the captured pointers (design T9), for every step 7 switch value."""
    q, k, v = _inputs(2, 4, 2, 300, 700, torch.bfloat16)
    api, o, lse = _plan(q, k, v, stats=True, is_causal=True, causal_bottom_right=True, variant=variant)

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
            assert torch.equal(o, eager[0]) and torch.equal(lse, eager[1]), (variant, "changed inputs" if step else "same inputs")
    finally:
        graph.reset()


# Compiles four plans, then runs each three times under a faulthandler deadline: a ring that never completes makes
# the child exit instead of hanging the worker (test/AGENTS.md, crash isolation), and the parent reports HANG.
# Self-contained, so it needs nothing from pytest's path setup.
_TURNS_CHILD = textwrap.dedent(r"""
    import dataclasses, faulthandler, sys, time
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    overlap, run_budget_s = sys.argv[1] == "1", int(sys.argv[2])
    # (b, h_q, h_kv, s_q, s_kv, causal): ring wrap; one KV tile (the prologue meets the drain); WG2 without a row
    # over three tiles of an incomplete Q tile (phase 3); bottom-right causal with masked and unmasked phases.
    CASES = [(1, 2, 2, 256, 384, False), (1, 2, 2, 64, 128, False), (2, 2, 2, 17, 300, False), (2, 4, 2, 130, 700, True)]
    plans = []
    for b, h_q, h_kv, s_q, s_kv, causal in CASES:
        torch.manual_seed(0)
        q, k, v = (torch.randn(b, s, h, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for s, h in ((s_q, h_q), (s_kv, h_kv), (s_kv, h_kv)))
        o = torch.full((b, s_q, h_q, 128), float("nan"), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        api = SdpaFwdDslSm90(q, k, v, o, **(dict(is_causal=True, causal_bottom_right=True) if causal else {}))
        assert api.check_support() and api.flavor == (128, 128), api.flavor
        api.params = dataclasses.replace(api.params, intra_wg_overlap=overlap)
        api.compile()
        assert api._k_mod.INTRA_WG_OVERLAP == overlap
        plans.append((api, q, k, v, o, (b, h_q, h_kv, s_q, s_kv, causal)))
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
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_d128_turns_and_rings_complete_in_a_watchdogged_child(variant):
    """Every launch of every switch value finishes: no K or V ring slot waits for an arrival that never comes,
    including one KV tile, a warpgroup without rows and an incomplete Q tile. A hang is a child that misses its
    run deadline, reported as HANG with its last lines; a child that stalls before every plan has compiled is
    reported as STALL, with its stack (the child arms its own deadline before its first import)."""
    argv = [sys.executable, "-c", _TURNS_PREAMBLE + _TURNS_CHILD, str(int(_VARIANTS[variant])), str(_TURNS_RUN_BUDGET_S), str(_TURNS_COMPILE_BUDGET_S)]
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
    """check_support writes api_dsl's D128 switch constant into a D128 plan's record at call time, and a
    D512 plan (tile 64/64 pinned) never carries it."""
    from cudnn.sdpa.fwd import api_dsl

    q, k, v = _inputs(1, 2, 2, 64, 128, torch.float16)
    o = torch.empty_like(q)
    for overlap in _VARIANTS.values():
        monkeypatch.setattr(api_dsl, "D128_INTRA_WG_OVERLAP", overlap)
        api = api_dsl.SdpaFwdDslSm90(q, k, v, o)
        assert api.check_support() and api.flavor == (128, 128), api.flavor
        assert api.params.intra_wg_overlap == overlap
        d512 = api_dsl.SdpaFwdDslSm90(q, k, v, o, tile_m=64, tile_n=64)
        assert d512.check_support() and d512.flavor == (512, 512), d512.flavor
        assert d512.params.intra_wg_overlap is False


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
    dump, fields, ptxas, cands = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_ARCH"] = "sm_90a"  # unconditional: an inherited value would pin another target
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template
    from cudnn.sdpa.fwd.config_sm90 import TemplateParams
    # The heaviest D128 body: causal (three masked phases), GQA and Stats; ``fields`` adds the step 7 switch.
    params = dataclasses.replace(TemplateParams(dtype_qkv=DTYPE_BF16, causal=True, has_lse=True, qh_per_kh=4), **fields)
    mod = _load_kernel_template("sm90/prefill_d128_f16.py", params, tag="sass_sm90_d128")
    print("SWITCHES", int(mod.INTRA_WG_OVERLAP))
    b, h_q, h_kv, s = 2, 8, 2, 1024
    q, kv = (s * h_q * 128, 128, h_q * 128, 1), (s * h_kv * 128, 128, h_kv * 128, 1)
    mod.compile(b, h_q, h_kv, s, s, q, kv, kv, q, (h_q * s, s, 1))
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
    # 66 in the kernel mapper's ov build, plan O1). Printed for the A/B record, not pinned.
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
# A target, not yet a measurement: the plan expects no spills at the 240-register budget (design
# 3.5 estimates about 200). Replace it with the STL/LDL counts of the first sm_90a compile (S2).
_SPILL_PIN = {"STL": 0, "LDL": 0}
# Per-value structure of the probe body (causal, GQA 4, Stats, bf16), measured on the offline sm_90a kit (ptxas 13.4
# V13.4.92, CuTe DSL 4.8.0, compile_d128.sh --preset probe): v1 = 3 step instances x (8 SS + 8 RS); ov = prologue QK 8
# + 2 step instances x 16 + drain PV 8, one wait_group(1) per step instance. Body counts are a toolchain fact
# (python/cudnn/sdpa/AGENTS.md, the LDTM lesson): re-measure on a new toolchain and record it next to the numbers.
_CODEGEN_PINS = {
    "v1": dict(HGMMA=48, DEPBAR=6, DEPBAR_1=0, BAR_ARV=0),
    "ov": dict(HGMMA=48, DEPBAR=6, DEPBAR_1=2, BAR_ARV=0),
}


def _ptxas():
    import os
    import shutil

    cands = [os.path.join(os.environ["CUDA_PATH"], "bin", "ptxas")] if os.environ.get("CUDA_PATH") else []
    cands.append(shutil.which("ptxas") or "")
    return next((c for c in cands if c and os.path.isfile(c) and os.access(c, os.X_OK)), None)


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_d128_register_split_no_spills_and_no_wgmma_serialization(tmp_path, variant):
    """The (24, 240, 240) setmaxnreg split survives into SASS (min_blocks_per_mp=1 on the launch;
    python/cudnn/AGENTS.md: ptxas otherwise drops it silently), the compute body does not spill and stays
    under the 240-register budget, and ptxas keeps the QK and PV WGMMA groups pipelined (no C7511/C7513
    serialization advisory), for every step 7 switch value. The switch must really compile: a
    wait_group(1) appears exactly under overlap."""
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / "dump"
    dump.mkdir()
    overlap = _VARIANTS[variant]
    fields = json.dumps(dict(intra_wg_overlap=overlap))
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, str(dump), fields, _ptxas() or "-", *cands], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"sm_90a trace-compile of the D128 template ({variant}) failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    assert f"SWITCHES {int(overlap)}" in out, [ln for ln in out if ln.startswith("SWITCHES")]
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith(("SASS ", "PTX "))}
    print(f"\nD128 SASS {variant}:", stats, [ln for ln in out if ln.startswith("PTXAS")])
    assert stats["USETMAXREG"] > 0, "no USETMAXREG: ptxas dropped the register split (C7508)"
    assert (stats["USETMAXREG_ALLOC"], stats["USETMAXREG_DEALLOC"]) == (240, 24), stats
    assert stats["HGMMA"] > 0, stats
    assert_no_new_spills(stats, _SPILL_PIN, tag=f"D128 sm_90a {variant}: ")
    assert stats["MAXREG"] < 240, f"{variant}: register R{stats['MAXREG']} is past the 240-register budget"
    assert (stats["DEPBAR_1"] > 0) == overlap, f"{variant}: wait_group(1) count {stats['DEPBAR_1']}"
    assert {key: stats[key] for key in _CODEGEN_PINS[variant]} == _CODEGEN_PINS[variant], (variant, stats)
    ptxas = {ln.split()[1]: ln.split()[2] for ln in out if ln.startswith("PTXAS ") and len(ln.split()) == 3}
    if "serialized" in ptxas:
        assert ptxas["rc"] == "0" and ptxas["serialized"] == "0", [ln for ln in out if ln.startswith("PTXAS")]
