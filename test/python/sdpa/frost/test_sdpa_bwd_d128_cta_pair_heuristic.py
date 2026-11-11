# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-S heuristic that routes the BF16 grouped d128 backward (packed THD and dense compact BSHD) between the one-CTA bodies and the
cta_group::2 pair body.

The typed adapter field ``SdpaBwdDslSm100D128.cta_pair`` ("auto" = the default: the per-S rule | "off" = the one-CTA body always |
"on" = the pair body, declining BY NAME when a predicate fails) is resolved by ``prepared_sm100_d128_grouped_2cta`` from ONE set of
config constants in ``d128_grouped_2cta_contract`` (``CTA_PAIR_MIN_SEQ`` = PER PART the smallest S where the pair body won the in-session
step: measured on cc 10.0, carried to cc 10.3 and cc 10.7; ``CTA_PAIR_CCS`` = its keys, cc 10.0 / 10.3 / 10.7 (the same tcgen05 / cta_group::2 / TMEM / CLC features; the threshold
was measured on GB200); the cluster-of-two admission through the DSL's occupancy query) plus every
support predicate of the pair body.  The host rows need no GPU: the pure selector, the typed domain, the declines by name, the
dispatch with its fallback, the resolution token.  The GPU rows (skipped off SM100 / SM103) pin the launch census per selection:
five launches either way on THD (four on the one-CTA dense body), the pair's cluster dims, the one-CTA kernel below the threshold and
under ``off``.

The DENSE twin (``d128_grouped_2cta_contract.LAYOUT_FORMS`` "dense"): a dense compact-BSHD BF16 GQA / MQA declaration is served by the pair
body under the same rule -- the selector's ``layout`` predicate admits "thd" and "dense" -- through the THD chain with synthesized metadata
(the contract module's docstring); the dense rows below pin the selection, the recorded dense form, the launch census and the twin property
(dK / dV BITWISE the THD chain's on the same per-sequence data: the same kernels in the same partial order; dQ by the suite's fp64 gate).

RED-first: every row below was written before the field (and the dense form) existed.
"""

import pathlib
import re
from types import SimpleNamespace

import pytest
import torch

from d128_grouped_test_utils import adapter_arguments, check_case, make_adapter, make_case, make_workspace, poison
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_thd_eligibility import compiler_version  # noqa: F401  (a fixture: the DSL floor modelled host-side)

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm100_d128"
_ONE_CTA_KERNEL = "bprop_d128_accum_grouped_thd_kernel"
_ONE_CTA_DENSE_KERNEL = "bprop_d128_accum_grouped_dense_kernel"
_ONE_CTA_IMPLEMENTATIONS = {"thd": "accumulation_grouped_thd", "dense": "accumulation_grouped_dense"}
_PAIR_KERNEL = "bprop_d128_accum_grouped_2cta_stail_kernel"
_SELECTION_TABLE = [  # S_max of the batch -> the body ``auto`` selects at cc 10.0 with a cluster of two admitted (the measured rule)
    pytest.param(2048, "1cta", id="s2k"),
    pytest.param(4096, "1cta", id="s4k"),
    pytest.param(8191, "1cta", id="s8k-1"),
    pytest.param(8192, "2cta", id="s8k"),
    pytest.param(16384, "2cta", id="s16k"),
    pytest.param(32768, "2cta", id="s32k"),
]
_BATCHES = [pytest.param("uniform", id="uniform"), pytest.param("ramp", id="ramp")]


def _contract():
    from cudnn.sdpa.bwd import d128_grouped_2cta_contract as contract

    return contract


def _prepared():
    from cudnn.sdpa.bwd import prepared_sm100_d128_grouped_2cta as prepared

    return prepared


def _lengths(s_max, form):
    """A THD batch whose LONGEST sequence is ``s_max``: uniform = four copies; ramp = four steps up to it (the heuristic reads the max)."""
    if form == "uniform":
        return (s_max,) * 4
    return tuple(s_max * k // 4 for k in (1, 2, 3, 4))


def _select(mode, s_max, form, **over):
    c = _contract()
    lens = _lengths(s_max, form)
    kw = dict(layout="thd", s_q_max=max(lens), s_kv_max=max(lens), cc=(10, 0), max_active_pairs=76)
    kw.update(over)
    return c.select_cta_pair(mode, **kw)


# ------------------------------------------------------------------------------------------------------------ host (no GPU)
def test_cta_pair_config_is_the_truth():
    """Every number the heuristic reads is ONE named constant of the contract module (G10 config-as-truth): the typed domain and its
    default, the S threshold (a multiple of the pair's kv unit and of the q tile), the measured part, the cluster width = the pair body's
    CGA_M, the pair body's identity = the pair body's TMEM-map name, the two body tokens."""
    c = _contract()
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import CFG
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG as STAIL_CFG
    from cudnn.sdpa.bwd.d128_contract import ACCUMULATION_CCS

    assert c.CTA_PAIR_MODES == ("auto", "off", "on") and c.CTA_PAIR_MODE_DEFAULT == "auto"
    # the threshold is PART-KEYED (cc -> S): the GB200 value on every admitted part today (cc 10.3 and cc 10.7 carry it until re-measured);
    # the admitted parts are the mapping's keys, every one an accumulation part, and an unadmitted cc has no threshold (ValueError by name)
    assert c.CTA_PAIR_CCS == ((10, 0), (10, 3), (10, 7)) and tuple(c.CTA_PAIR_MIN_SEQ) == c.CTA_PAIR_CCS
    assert all(tuple(cc) in ACCUMULATION_CCS for cc in c.CTA_PAIR_CCS)
    for cc in c.CTA_PAIR_CCS:
        seq = c.cta_pair_min_seq(cc)
        assert seq == c.CTA_PAIR_MIN_SEQ[cc] == 8192 and seq % c.UNIT_ROWS == 0 and seq % CFG.PAIR_KV_ROWS == 0, cc
    for cc in ((8, 0), (12, 0), None):
        with pytest.raises(ValueError, match="no cta_pair threshold"):
            c.cta_pair_min_seq(cc)
    assert c.CTA_PAIR_CLUSTER == CFG.CGA_M == 2
    assert c.CTA_PAIR_BODY == "s_tail" == STAIL_CFG.TMEM_MAP
    assert (c.PAIR_BODY, c.ONE_CTA_BODY) == ("2cta", "1cta")
    from cudnn.sdpa.bwd.d128_grouped_contract import DSL_MIN_VERSION, dsl_version_decline

    assert DSL_MIN_VERSION == (4, 8, 0) and dsl_version_decline(True, ("nvidia-cutlass-dsl", "4.8.0")) is None
    assert dsl_version_decline(False, None) == dsl_version_decline(True, None) == "requires nvidia-cutlass-dsl >= 4.8.0; found None"
    assert dsl_version_decline(True, ("nvidia-cutlass-dsl", "4.7.0")).startswith("requires nvidia-cutlass-dsl >= 4.8.0; found ")
    assert dsl_version_decline(True, ("nvidia-cutlass-dsl-internal", "0.3.0+2026")) is not None  # an internal build: not the measured public floor
    source = pathlib.Path(c.__file__).read_text()  # the threshold's provenance sentence sits beside the constant (the module docstring is in it)
    assert "smallest S where the pair body won" in source


def test_cta_pair_field_is_typed_and_the_record_defaults_are_pinned():
    """``cta_pair`` is a typed op-level field: ``None`` resolves to the default ``auto``; anything outside the domain declines BY NAME (a
    ValueError naming the field and the domain); the adapter reader prefixes the engine name; the shipping class defaults to ``auto``, the
    direct 2-CTA entry pins ``on``; no class carries a bool body selector (``datapath_2cta``)."""
    c = _contract()
    p = _prepared()
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.api_dsl_sm100_d128_2cta import SdpaBwdDslSm100D128Cta2

    assert c.resolve_cta_pair(None) == "auto"
    for mode in c.CTA_PAIR_MODES:
        assert c.resolve_cta_pair(mode) == mode
    for bad in ("bogus", True, False, 1, "", "AUTO", "On", "pair", "2cta"):
        with pytest.raises(ValueError, match="cta_pair must be one of"):
            c.resolve_cta_pair(bad)
    assert p.cta_pair(SimpleNamespace(_NAME=_ENGINE)) == "auto" and p.cta_pair(SimpleNamespace(_NAME=_ENGINE, cta_pair="off")) == "off"
    with pytest.raises(ValueError, match=f"{_ENGINE}: cta_pair must be one of"):
        p.cta_pair(SimpleNamespace(_NAME=_ENGINE, cta_pair=True))
    assert SdpaBwdDslSm100D128.cta_pair == "auto" and SdpaBwdDslSm100D128Cta2.cta_pair == "on"
    assert not hasattr(SdpaBwdDslSm100D128, "datapath_2cta") and not hasattr(SdpaBwdDslSm100D128Cta2, "datapath_2cta")
    assert not hasattr(p, "datapath_2cta"), "the bool reader is folded into cta_pair_resolution"


@pytest.mark.parametrize("s_max, body", _SELECTION_TABLE)
@pytest.mark.parametrize("form", _BATCHES)
def test_cta_pair_auto_selects_by_the_longest_sequence(s_max, body, form):
    """``auto`` at cc 10.0 with a cluster of two admitted: the pair body at S_max >= CTA_PAIR_MIN_SEQ, the one-CTA body below -- for a uniform
    batch and for a ramp whose LONGEST sequence is S_max (the heuristic reads the batch's max sequence length, both q and kv)."""
    c = _contract()
    res = _select("auto", s_max, form)
    assert res.body == body, res
    assert res.mode == "auto" and res.token == f"pair=auto:{body}"
    assert res.s_q_max == res.s_kv_max == s_max
    if body == c.PAIR_BODY:
        assert res.cluster_admitted is True and str(c.cta_pair_min_seq((10, 0))) in res.reason
    else:
        assert "seq" in res.reason and str(c.cta_pair_min_seq((10, 0))) in res.reason


def test_cta_pair_auto_reads_both_q_and_kv_extents():
    """A lopsided batch (one side below the threshold) falls to the measured-safe one-CTA body: the rule is the conjunction."""
    c = _contract()
    assert _select("auto", 16384, "uniform", s_kv_max=4096).body == c.ONE_CTA_BODY
    assert _select("auto", 16384, "uniform", s_q_max=4096).body == c.ONE_CTA_BODY
    assert _select("auto", 16384, "uniform", s_q_max=8192, s_kv_max=16384).body == c.PAIR_BODY


def test_cta_pair_auto_falls_back_by_named_predicate():
    """Each predicate the pair needs sends ``auto`` to the one-CTA body with the predicate NAMED in the reason: the layout (a form outside
    ``CTA_PAIR_LAYOUTS``: neither the packed chain nor its dense twin), the admitted parts (cc 10.0 / 10.3 / 10.7: Hopper, SM120 and a part of
    another line have no cta_group::2 pair), the cluster-of-two admission (0 active pairs from the occupancy query), a device that is not CUDA
    at all.  Both served layouts select the pair."""
    c = _contract()
    assert c.CTA_PAIR_LAYOUTS == c.LAYOUT_FORMS == ("thd", "dense")
    for layout in c.CTA_PAIR_LAYOUTS:
        res = _select("auto", 16384, "uniform", layout=layout)
        assert res.body == c.PAIR_BODY, (layout, res)
    for over, name in (
        (dict(layout="bshd_padded"), "layout"),
        (dict(layout=None), "layout"),
        (dict(cc=(8, 0)), "cc"),
        (dict(cc=(9, 0)), "cc"),
        (dict(cc=(12, 0)), "cc"),
        (dict(cc=None), "cc"),
        (dict(max_active_pairs=0), "cluster"),
    ):
        res = _select("auto", 16384, "uniform", **over)
        assert res.body == c.ONE_CTA_BODY and res.mode == "auto", (over, res)
        assert re.search(rf"\b{name}\b", res.reason), (over, res.reason)
    # the occupancy query is consulted only when every cheaper predicate passed (it compiles a probe kernel): a short S never calls it
    calls = []

    def probe():
        calls.append(1)
        return 76

    assert _select("auto", 2048, "uniform", max_active_pairs=probe).body == c.ONE_CTA_BODY and calls == []
    assert _select("auto", 16384, "uniform", cc=(9, 0), max_active_pairs=probe).body == c.ONE_CTA_BODY and calls == []
    assert _select("auto", 16384, "uniform", max_active_pairs=probe).body == c.PAIR_BODY and calls == [1]
    for cc in c.CTA_PAIR_CCS:  # every part of the SM100 line selects the pair under auto at the threshold (cc 10.3: the sm103 CI lane)
        res = _select("auto", 16384, "uniform", cc=cc, max_active_pairs=76)
        assert res.body == c.PAIR_BODY and res.cc == cc and f"cc {cc}" in res.reason, (cc, res)


def test_cta_pair_on_declines_by_name_and_ignores_the_s_rule():
    """``on`` forces the pair body at ANY S (the threshold is the heuristic's, not a support predicate) and declines BY NAME when a pair predicate
    fails: the decline text names the predicate (layout / cc / cluster) and the value seen.  Dense is a served layout: ``on`` selects the pair."""
    c = _contract()
    for s_max in (256, 2048, 8192, 32768):
        for layout in c.CTA_PAIR_LAYOUTS:
            res = _select("on", s_max, "uniform", layout=layout)
            assert res.body == c.PAIR_BODY and res.token == "pair=on:2cta", (layout, res)
    with pytest.raises(ValueError, match=r"cta_pair 'on' requires a layout in \('thd', 'dense'\).*got 'bshd_padded'"):
        _select("on", 16384, "uniform", layout="bshd_padded")
    for cc in ((8, 0), (9, 0), (11, 0), (12, 0)):
        with pytest.raises(ValueError, match=rf"cta_pair 'on' requires cc .*\(10, 0\), \(10, 3\), \(10, 7\).*got {re.escape(str(cc))}"):
            _select("on", 16384, "uniform", cc=cc)
    for cc in c.CTA_PAIR_CCS:
        assert _select("on", 256, "uniform", cc=cc).body == c.PAIR_BODY, cc
    with pytest.raises(ValueError, match=r"cta_pair 'on' requires cc .*got a device that is not CUDA"):
        _select("on", 16384, "uniform", cc=None)
    with pytest.raises(ValueError, match=r"cta_pair 'on' requires a cluster of 2 CTAs admitted .*got 0"):
        _select("on", 16384, "uniform", max_active_pairs=0)


def test_cta_pair_off_is_the_one_cta_body_at_every_s_without_probing():
    """``off`` never consults a device predicate: the one-CTA body at every S, on every part, with the probe never called."""
    c = _contract()

    def probe():
        raise AssertionError("off must not probe the device")

    for s_max in (256, 8192, 32768):
        for cc in ((10, 0), (10, 3), None):
            res = _select("off", s_max, "uniform", cc=cc, max_active_pairs=probe)
            assert res.body == c.ONE_CTA_BODY and res.token == "pair=off:1cta" and res.cluster_admitted is None, res


def test_cta_pair_selector_is_typed_on_its_inputs():
    with pytest.raises(ValueError, match="cta_pair must be one of"):
        _select("bogus", 8192, "uniform")
    for bad in (dict(s_q_max=0), dict(s_kv_max=-1), dict(s_q_max="8192"), dict(s_q_max=8192.0)):
        with pytest.raises(ValueError, match="sequence extents must be positive integers"):
            _select("auto", 8192, "uniform", **bad)


def _fake_api(mode, s=16384, thd=True, device="cpu"):
    """A declaration-only stand-in for the adapter: the fields the resolution reads and nothing else."""
    desc = SimpleNamespace(shape=(1, 8, s, 128), device=torch.device(device))
    api = SimpleNamespace(_NAME=_ENGINE, cta_pair=mode, thd=thd, q_desc=desc, k_desc=SimpleNamespace(shape=(1, 2, s, 128), device=desc.device))
    api._cta_pair = None
    api._uses_grouped_heads = lambda: True
    return api


def test_cta_pair_resolution_records_the_device_facts_on_the_instance(monkeypatch):
    """``resolve_cta_pair_body`` reads the declared S_q / S_kv, the device's cc (None off CUDA) and the cached occupancy admission, records the
    resolution on the adapter instance and returns it through ``cta_pair_resolution``; the token is ``pair=<mode>:<body>``."""
    c = _contract()
    p = _prepared()
    monkeypatch.setattr(p, "_device_cc", lambda api: (10, 0))
    monkeypatch.setattr(p, "max_active_pairs", lambda device: 76)
    api = _fake_api("auto")
    res = p.resolve_cta_pair_body(api)
    assert res.body == c.PAIR_BODY and res.cc == (10, 0) and res.cluster_admitted is True and res.s_q_max == res.s_kv_max == 16384
    assert p.cta_pair_resolution(api) is res and res.token == "pair=auto:2cta"
    assert p.resolve_cta_pair_body(_fake_api("auto", s=4096)).body == c.ONE_CTA_BODY
    assert p.cta_pair_resolution(_fake_api("off")) is None, "nothing recorded before the resolution runs"
    monkeypatch.setattr(p, "_device_cc", lambda api: None)
    with pytest.raises(NotImplementedError, match=f"{_ENGINE}: cta_pair 'on' requires cc"):
        p.resolve_cta_pair_body(_fake_api("on"))
    with pytest.raises(ValueError, match=f"{_ENGINE}: cta_pair must be one of"):
        p.resolve_cta_pair_body(_fake_api("bogus"))


def test_cta_pair_dispatch_falls_back_under_auto_and_reraises_under_on(monkeypatch):
    """The grouped dispatch: ``auto`` that selected the pair runs the pair body's own support predicates and, when one declines, serves the
    one-CTA body with the decline recorded in the reason; ``on`` re-raises that decline unchanged (by name); ``off`` never touches the pair
    path.  A pair accepted records the pair body's identity (``CTA_PAIR_BODY``) as ``pair_map`` on the resolution."""
    c = _contract()
    p = _prepared()
    monkeypatch.setattr(p, "_device_cc", lambda api: (10, 0))
    monkeypatch.setattr(p, "max_active_pairs", lambda device: 76)
    seen = []
    monkeypatch.setattr(p, "_check_support_grouped", lambda api: seen.append("1cta") or True)

    def pair_declines(api):
        seen.append("pair")
        raise NotImplementedError(f"{_ENGINE}: 2-CTA datapath requires something the test withholds")

    monkeypatch.setattr(p, "check_support", pair_declines)
    api = _fake_api("auto")
    assert p.check_support_selected(api) is True
    assert seen == ["pair", "1cta"] and api._cta_pair.body == c.ONE_CTA_BODY and "pair body declined" in api._cta_pair.reason
    assert "the test withholds" in api._cta_pair.reason and api._cta_pair.pair_map is None
    seen.clear()
    with pytest.raises(NotImplementedError, match="requires something the test withholds"):
        p.check_support_selected(_fake_api("on"))
    assert seen == ["pair"]
    seen.clear()
    assert p.check_support_selected(_fake_api("off")) is True and seen == ["1cta"]
    seen.clear()
    assert p.check_support_selected(_fake_api("auto", s=2048)) is True and seen == ["1cta"]
    # the pair accepted: the resolution records the pair body's identity under auto and under on

    def pair_accepts(api):
        seen.append("pair")
        return True

    monkeypatch.setattr(p, "check_support", pair_accepts)
    for mode in ("auto", "on"):
        seen.clear()
        api = _fake_api(mode)
        assert p.check_support_selected(api) is True and seen == ["pair"]
        assert api._cta_pair.body == c.PAIR_BODY and api._cta_pair.pair_map == c.CTA_PAIR_BODY == "s_tail"


def test_cta_pair_dispatch_gates_the_dsl_before_the_occupancy_probe(monkeypatch):
    """A missing or too-old CuTe DSL is the grouped chain's typed decline (NotImplementedError naming the floor and the installed version) at
    ``check_support``, BEFORE the dispatch runs the cluster occupancy probe: the probe imports the DSL (``cutlass.utils.HardwareInfo``), so an
    absent wheel raised a raw ModuleNotFoundError and an older one an AttributeError out of ``check_support()`` (PR 1943 review).  Under
    ``auto`` and ``on`` alike, with every cheaper predicate passing (S 16K, cc 10.0); ``off`` is gated the same way (its one-CTA body has the
    same floor)."""
    c = _contract()
    p = _prepared()
    import cudnn.frost.buffers as buffers

    def probe(device):
        raise probe.error  # what the occupancy query raises on the DSL state under test

    monkeypatch.setattr(p, "_device_cc", lambda api: (10, 0))
    monkeypatch.setattr(p, "max_active_pairs", probe)
    floor = ".".join(map(str, getattr(c, "DSL_MIN_VERSION", (4, 8, 0))))
    for state, error in (
        ((False, None), ModuleNotFoundError("No module named 'cutlass'")),
        ((True, ("nvidia-cutlass-dsl", "4.7.0")), AttributeError("'HardwareInfo' object has no attribute 'get_max_active_clusters'")),
    ):
        probe.error = error
        monkeypatch.setattr(buffers, "cutedsl_state", lambda state=state: state)
        for mode in ("auto", "on", "off"):
            api = _fake_api(mode)
            with pytest.raises(NotImplementedError, match=rf"{_ENGINE}: grouped d128 requires nvidia-cutlass-dsl >= {re.escape(floor)}; found "):
                p.check_support_selected(api)
            assert not getattr(api, "_is_supported", False)


def test_cta_pair_probe_failures_are_named_in_the_cluster_decline(monkeypatch):
    """The occupancy query's OWN failures (the DSL present and new enough; the query raising from inside the DSL or the driver) are never an
    exception out of ``check_support``: the device admits no pair, the ``cluster`` predicate declines by name with the query's text -- under
    ``on`` the decline, under ``auto`` the one-CTA body with the text in the reason; the answer is cached per device like a successful query."""
    c = _contract()
    p = _prepared()
    import cudnn.frost.device as frost_device

    monkeypatch.setattr(p, "_device_cc", lambda api: (10, 0))
    monkeypatch.setattr(frost_device, "resolve_device", lambda device=None: 0)  # host-only: the index read of the stand-in's CUDA device
    monkeypatch.setattr(p, "_check_support_grouped", lambda api: True)
    for error in (RuntimeError("cuOccupancyMaxActiveClusters failed"), ImportError("No module named 'cutlass.utils'"), AttributeError("no query")):
        p._MAX_ACTIVE_PAIRS.clear()
        p._PROBE_FAILURES.clear()
        seen = []

        def query(index, error=error):
            seen.append(index)
            raise error

        monkeypatch.setattr(p, "_occupancy_query", query)
        api = _fake_api("auto", device="cuda")
        assert p.check_support_selected(api) is True
        assert api._cta_pair.body == c.ONE_CTA_BODY and api._cta_pair.cluster_admitted is False, api._cta_pair
        assert "cluster" in api._cta_pair.reason and f"{type(error).__name__}: {error}" in api._cta_pair.reason, api._cta_pair.reason
        with pytest.raises(NotImplementedError, match=rf"{_ENGINE}: cta_pair 'on' requires a cluster of .*{type(error).__name__}: {re.escape(str(error))}"):
            p.check_support_selected(_fake_api("on", device="cuda"))
        assert len(seen) == 1, "the failed query is cached per device like a successful one"
    p._MAX_ACTIVE_PAIRS.clear()
    p._PROBE_FAILURES.clear()


def test_cta_pair_support_validates_the_pair_fields_by_name(monkeypatch):
    """The pair body's ``check_support`` resolves EVERY typed field of the body (``mask_form``, ``warp_roles``, ``diag_2cta``, ``debug_clk``,
    ``dkv_accum``) and the role shape's cross-field rule, so a value outside a domain is a ValueError BY NAME and an unbuilt combination a typed
    decline at SUPPORT time -- never a ValueError out of ``compile()`` after ``auto`` committed to the pair (PR 1943 review).  Under ``auto`` the
    unbuilt combination falls back to the one-CTA body with the decline in the reason; under ``on`` it is the answer; the served bundle passes."""
    c = _contract()
    p = _prepared()
    monkeypatch.setattr(p, "_device_cc", lambda api: (10, 0))
    monkeypatch.setattr(p, "max_active_pairs", lambda device: 76)

    def grouped_ok(api):  # the one-CTA route's checks, stubbed: the pair check reads the declaration facts they record
        s = int(api.q_desc.shape[2])
        api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max = 1, 8, 2, s, s
        api._thd_t_q = api._thd_t_kv = api._thd_t_q_phys = api._thd_t_kv_phys = s
        return True

    monkeypatch.setattr(p, "_check_support_grouped", grouped_ok)
    for field, bad, text in (
        ("warp_roles", "bogus", "warp_roles must be one of"),
        ("mask_form", "bogus", "mask_form must be one of"),
        ("diag_2cta", "ds_gates_off", "diag_2cta must be None or one of"),
        ("debug_clk", "yes", "debug_clk must be a bool"),
        ("dkv_accum", "fp16", "dkv_accum must be one of"),
    ):
        for mode in ("auto", "on"):
            api = _fake_api(mode)
            setattr(api, field, bad)
            with pytest.raises(ValueError, match=f"{_ENGINE}: {text}"):
                p.check_support_selected(api)
            assert not getattr(api, "_is_supported", False), (field, mode)
    unbuilt = "debug_clk is not defined under warp_roles '16w_fa4'"
    api = _fake_api("on")
    api.warp_roles, api.debug_clk = "16w_fa4", True
    with pytest.raises(NotImplementedError, match=f"{_ENGINE}: 2-CTA datapath {unbuilt}"):
        p.check_support_selected(api)
    api = _fake_api("auto")
    api.warp_roles, api.debug_clk = "16w_fa4", True
    assert p.check_support_selected(api) is True
    assert api._cta_pair.body == c.ONE_CTA_BODY and unbuilt in api._cta_pair.reason, api._cta_pair
    api = _fake_api("auto")
    api.warp_roles, api.dkv_accum, api.mask_form = "16w_fa4", "bf16_partials", "predicated"
    assert p.check_support_selected(api) is True and api._cta_pair.body == c.PAIR_BODY and api._is_supported is True


def _grouped_ok(api):
    """The one-CTA route's checks, stubbed: the pair check reads the declaration facts they record."""
    s = int(api.q_desc.shape[2])
    api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max = 1, 8, 2, s, s
    api._thd_t_q = api._thd_t_kv = api._thd_t_q_phys = api._thd_t_kv_phys = s
    return True


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)], ids=["cc10.0", "cc10.3", "cc10.7"])
def test_cta_pair_admits_every_part_of_the_sm100_line(monkeypatch, cc):
    """RED-first (the sm103 CI lane, cc 10.3, failed every pair-body row with the typed decline; cc 10.7 added the same way): with the device cc
    read as ``cc`` the pair body is SELECTED under ``auto`` at that part's threshold and ``on`` passes ``check_support`` at any S; the resolution
    records the cc.  The compile target follows the device the same way as the one-CTA body (``--gpu-arch sm_{{cc}}a`` from the device cc in
    both prepared hosts), so nothing else is per part."""
    c = _contract()
    p = _prepared()
    monkeypatch.setattr(p, "_device_cc", lambda api: cc)
    monkeypatch.setattr(p, "max_active_pairs", lambda device: 76)
    monkeypatch.setattr(p, "_check_support_grouped", _grouped_ok)
    api = _fake_api("auto", s=c.cta_pair_min_seq(cc))
    assert p.check_support_selected(api) is True and api._is_supported is True
    assert api._cta_pair.body == c.PAIR_BODY and api._cta_pair.cc == cc and api._cta_pair.token == "pair=auto:2cta", api._cta_pair
    api = _fake_api("auto", s=c.cta_pair_min_seq(cc) - 256)  # one pair unit below that part's threshold: the one-CTA body, the seq predicate named
    assert p.check_support_selected(api) is True and api._cta_pair.body == c.ONE_CTA_BODY and "seq" in api._cta_pair.reason, api._cta_pair
    api = _fake_api("on", s=256)
    assert p.check_support_selected(api) is True and api._cta_pair.body == c.PAIR_BODY and api._cta_pair.cc == cc, api._cta_pair


@pytest.mark.parametrize("cc", [(8, 0), (11, 0), (12, 0)], ids=["cc8.0", "cc11.0", "cc12.0"])
def test_pair_body_helper_skips_on_a_typed_cc_decline(monkeypatch, cc):
    """RED-first (the sm80 / sm120 CI lanes): the suite's pair-body construction (``d128_2cta_test_utils.make_adapter_2cta``) maps the library's
    typed decline (``NotImplementedError`` out of ``check_support``, here the cc predicate) to a pytest SKIP whose reason is the decline text --
    never a failure; the library's behaviour is unchanged (the decline itself is pinned by ``test_cta_pair_on_declines_by_name_...``)."""
    from d128_2cta_test_utils import make_adapter_2cta

    p = _prepared()
    monkeypatch.setattr(p, "_device_cc", lambda api: cc)
    with pytest.raises(pytest.skip.Exception, match=rf"pair body declined on this device: .*cta_pair 'on' requires cc .*got {re.escape(str(cc))}"):
        make_adapter_2cta(_cpu_case())


def _cpu_case(s=256, lens=None):
    lens = lens or (s, s)
    return make_case(lens, lens, hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, device="cpu")


def test_cta_pair_real_adapter_declines_typed_off_cuda():
    """The shipping adapter on CPU descriptors: a bad ``cta_pair`` declines BY NAME before any support check; ``on`` declines naming the cc
    predicate (no CUDA device); ``off`` and ``auto`` reach the one-CTA grouped THD path, whose own decline names Q's device -- so the dispatch
    went to the one-CTA body, never into the pair path."""
    case = _cpu_case()
    api = make_adapter(case, compile_plan=False)
    assert type(api).cta_pair == "auto"
    api.cta_pair = "pair"
    with pytest.raises(ValueError, match=f"{_ENGINE}: cta_pair must be one of"):
        api.check_support()
    api.cta_pair = "on"
    with pytest.raises(NotImplementedError, match=f"{_ENGINE}: cta_pair 'on' requires cc .*not CUDA"):
        api.check_support()
    for mode in ("off", "auto"):
        api = make_adapter(case, compile_plan=False)
        api.cta_pair = mode
        with pytest.raises(NotImplementedError, match="grouped d128 Q must reside on CUDA"):
            api.check_support()
        assert api._cta_pair.body == "1cta" and api._cta_pair.mode == mode


def test_cta_pair_on_requires_grouped_heads_typed():
    """``on`` on an MHA declaration declines by name (the MHA path has no pair body) -- through the direct 2-CTA entry, whose class pins ``on``."""
    from d128_thd_test_utils import declaration as mha_declaration
    from d128_thd_test_utils import make_case as make_mha_case
    from d128_2cta_test_utils import make_adapter_2cta

    case = make_mha_case((129, 3), (129, 3), heads=2, device="cpu")
    api = make_adapter_2cta(case, compile_plan=False, declaration=mha_declaration)
    assert type(api).cta_pair == "on"
    with pytest.raises(NotImplementedError, match=f"{_ENGINE}: cta_pair 'on' requires grouped heads"):
        api.check_support()


def test_cta_pair_token_and_prose_name_the_predicates():
    """The decline texts and the reasons are stable tokens a driver pins by value (R4): the predicate names are the ones the tracker row lists."""
    c = _contract()
    assert c.CTA_PAIR_PREDICATES == ("layout", "cc", "cluster", "seq")
    res = _select("auto", 2048, "uniform")
    assert res.reason.startswith("cta_pair=auto: the one-CTA body (seq:")
    res = _select("auto", 8192, "uniform")
    assert res.reason.startswith("cta_pair=auto: the pair body (")


def test_cta_pair_dense_layout_form_is_typed():
    """The dense twin's typed plumbing (config = truth): the layout form domain and its Constexpr codes, the derivation from the ``thd`` flag,
    the implementation token per layout, the synthesized stats origin spec, the packed-view capacities of a dense declaration -- and the setup
    kernel's arm constants read from the same table."""
    c = _contract()
    import importlib

    assert c.LAYOUT_FORMS == ("thd", "dense") and c.LAYOUT_FORM_CODES == {"thd": 0, "dense": 1}
    assert c.IMPLEMENTATIONS == {"thd": c.THD, "dense": c.DENSE} and c.DENSE == "accumulation_grouped_dense_2cta"
    assert (c.layout_form_of(True), c.layout_form_of(False)) == ("thd", "dense")
    for name in c.LAYOUT_FORMS:
        assert c.resolve_layout_form(name) == name and c.layout_form_code(name) == c.LAYOUT_FORM_CODES[name]
    for bad in (None, True, "THD", "bshd", 1, ""):
        with pytest.raises(ValueError, match=r"layout must be one of \('thd', 'dense'\)"):
            c.resolve_layout_form(bad)
    assert c.DENSE_STATS_ORIGIN_SPEC == (8, 1, 1) and c.DENSE_ORIGIN_SPECS == (None,) * 8 + ((8, 1, 1),) and c.DENSE_ORIGIN_DTYPES == (None,) * 9
    assert c.dense_capacities(4, 8192, 16384) == (4 * 8192, 4 * 16384, 4 * 8192, 4 * 16384)
    with pytest.raises(ValueError, match="positive integer"):
        c.dense_capacities(0, 8192, 8192)
    helpers = importlib.import_module("cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16")  # the pair chain kernels: section 9 of the family module
    assert (helpers.LAYOUT_FORM_THD, helpers.LAYOUT_FORM_DENSE) == (c.LAYOUT_FORM_CODES["thd"], c.LAYOUT_FORM_CODES["dense"])
    host = importlib.import_module("cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host")
    import cutlass

    with pytest.raises(ValueError, match="layout must be one of"):
        host.compile_host_grouped_2cta((1, 8, 2, False), c.DENSE_ORIGIN_SPECS, c.DENSE_ORIGIN_DTYPES, cutlass.BFloat16, True, 100, layout="bshd")
    with pytest.raises(ValueError, match="layout 'dense' binds head-major Stats with the synthesized stats origin only"):
        host.compile_host_grouped_2cta((1, 8, 2, True), c.DENSE_ORIGIN_SPECS, c.DENSE_ORIGIN_DTYPES, cutlass.BFloat16, True, 100, layout="dense")
    with pytest.raises(ValueError, match="layout 'dense' binds head-major Stats with the synthesized stats origin only"):
        host.compile_host_grouped_2cta((1, 8, 2, False), (None,) * 9, c.DENSE_ORIGIN_DTYPES, cutlass.BFloat16, True, 100, layout="dense")


def test_cta_pair_dense_dispatch_selects_the_pair_and_records_the_dense_form(monkeypatch):
    """A dense declaration (``thd`` False) at S >= ``CTA_PAIR_MIN_SEQ[cc]`` on cc 10.0 with a cluster admitted: ``auto`` selects the pair, the pair's
    support records the dense form -- the packed-view capacities B*S on both sides, the Stats pitch S_q, the synthesized stats origin spec,
    no origin dtypes, the dense implementation token -- and the workspace is the THD chain's at those capacities; below the threshold the dense
    one-CTA body serves; ``on`` selects the pair at any S."""
    c = _contract()
    p = _prepared()
    monkeypatch.setattr(p, "_device_cc", lambda api: (10, 0))
    monkeypatch.setattr(p, "max_active_pairs", lambda device: 76)
    b, h, hk = 2, 8, 2

    def grouped_ok(api):  # the one-CTA route's dense checks, stubbed: the pair check reads the declaration facts they record
        s = int(api.q_desc.shape[2])
        api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max = b, h, hk, s, s
        return True

    monkeypatch.setattr(p, "_check_support_grouped", grouped_ok)
    for mode, s, body in (("auto", 16384, c.PAIR_BODY), ("auto", 4096, c.ONE_CTA_BODY), ("on", 256, c.PAIR_BODY), ("off", 16384, c.ONE_CTA_BODY)):
        api = _fake_api(mode, s=s, thd=False)
        assert p.check_support_selected(api) is True and api._cta_pair.body == body, (mode, s, api._cta_pair)
        if body == c.PAIR_BODY:
            assert api._implementation == c.DENSE and api._cta_pair.pair_map == c.CTA_PAIR_BODY
            assert (api._thd_t_q, api._thd_t_kv, api._thd_t_q_phys, api._thd_t_kv_phys) == (b * s, b * s, b * s, b * s)
            assert api._thd_head_pitch == s and api._thd_origins == c.DENSE_ORIGIN_SPECS and api._thd_origin_dtypes == c.DENSE_ORIGIN_DTYPES
            assert (api._grouped_workspace_regions, api._grouped_workspace_bytes) == c.thd_workspace_layout(b, h, b * s, b * s, True)
        else:
            assert not hasattr(api, "_implementation") or api._implementation != c.DENSE
    monkeypatch.setattr(p, "_check_support_grouped", _grouped_ok)  # the packed route's stub records the packed capacities itself
    api = _fake_api("auto", s=16384, thd=True)
    assert p.check_support_selected(api) is True and api._implementation == c.THD, "the packed form keeps its token"


# ------------------------------------------------------------------------------------------------------------ host: the REAL adapter on dense declarations
_ONE_CTA_DENSE = _ONE_CTA_IMPLEMENTATIONS["dense"]


def _declaration_api(monkeypatch, geometry, *, dtype=torch.bfloat16, causal=True, cc=(10, 0), mode=None, **fields):
    """The shipping adapter on declared geometry alone (``TensorDesc`` on a CUDA device, no tensors: the eligibility suite's recipe) with the device
    facts the dispatch and the grouped checks read modelled host-side -- the device cc (both the pair's reader and the grouped checks'), the
    occupancy admission, the DSL floor, the binder -- so the REAL ``check_support`` runs every grouped dense check and the pair's dense branch on
    a login host.  ``geometry`` = {role: (shape, stride)} over the nine ports; ``mode`` sets ``cta_pair``; ``fields`` any other typed field."""
    from cudnn import _pybind_module
    from cudnn.api_base import TensorDesc
    from cudnn.frost import buffers, device
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from d128_thd_test_utils import SCALE

    p = _prepared()
    monkeypatch.setattr(p, "_device_cc", lambda api: cc)
    monkeypatch.setattr(p, "max_active_pairs", lambda device: 76)
    monkeypatch.setattr(device, "compute_capability", lambda _: cc)
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))
    samples = {
        "sample_"
        + role: TensorDesc(
            dtype=torch.float32 if role == "stats" else dtype,
            shape=tuple(shape),
            stride=tuple(stride),
            stride_order=TensorDesc._compute_stride_order(tuple(shape), tuple(stride)),
            device=torch.device("cuda", 0),
            name=role,
        )
        for role, (shape, stride) in geometry.items()
    }
    api = SdpaBwdDslSm100D128(**samples, is_causal=causal, scale_softmax=SCALE, thd=False)
    if mode is not None:
        api.cta_pair = mode
    for name, value in fields.items():
        setattr(api, name, value)
    return api


def _dense_geometry(*, b=2, hq=4, hkv=2, s=256, pad_batch=False):
    """A dense compact-BSHD GQA declaration (B, H, S, 128) strides (S*H*128, 128, H*128, 1) with contiguous Stats (B, H, S, 1); ``pad_batch`` declares
    a batch stride twice the compact one (a padded, non-compact BSHD layout)."""
    from d128_thd_test_utils import Q_ROLES, ROLES

    geometry = {}
    for role in ROLES:
        heads = hq if role in Q_ROLES else hkv
        if role == "stats":
            geometry[role] = (b, hq, s, 1), (hq * s, s, 1, 1)
        else:
            geometry[role] = (b, heads, s, 128), (s * heads * 128 * (2 if pad_batch else 1), 128, heads * 128, 1)
    return geometry


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)], ids=["cc10.0", "cc10.3", "cc10.7"])
def test_cta_pair_dense_declaration_dispatches_through_the_real_checks(monkeypatch, cc):
    """RED-first (the pair body declined every dense declaration "requires THD"): the shipping adapter's REAL ``check_support`` on a dense compact-BSHD
    BF16 GQA declaration -- every grouped dense check live (compact strides, contiguous Stats, S % 128, BF16, the bounds), nothing stubbed -- on
    every admitted part (``CTA_PAIR_CCS``): ``auto`` at S >= ``CTA_PAIR_MIN_SEQ[cc]`` resolves the pair body with the dense implementation token and the THD chain's
    workspace at the packed-view capacities B*S; below the threshold and under ``off`` the one-CTA grouped dense body with its own token and
    workspace; ``on`` the pair body at any S."""
    from cudnn.sdpa.bwd.d128_grouped_contract import dense_workspace_layout

    c = _contract()
    b, hq, hkv = 2, 4, 2
    for mode, s, body, implementation in (
        ("auto", c.cta_pair_min_seq(cc), c.PAIR_BODY, c.DENSE),
        ("auto", 256, c.ONE_CTA_BODY, _ONE_CTA_DENSE),
        ("on", 256, c.PAIR_BODY, c.DENSE),
        ("on", c.cta_pair_min_seq(cc), c.PAIR_BODY, c.DENSE),
        ("off", c.cta_pair_min_seq(cc), c.ONE_CTA_BODY, _ONE_CTA_DENSE),
    ):
        api = _declaration_api(monkeypatch, _dense_geometry(b=b, hq=hq, hkv=hkv, s=s), cc=cc, mode=mode)
        assert api.check_support() is True and api._is_supported is True, (mode, s)
        res = api._cta_pair
        assert (res.body, res.token, res.mode) == (body, f"pair={mode}:{body}", mode), (mode, s, res)
        assert api._implementation == implementation, (mode, s, api._implementation)
        if body == c.PAIR_BODY:
            assert res.cc == cc and res.pair_map == c.CTA_PAIR_BODY, res
            assert api.scratch_workspace_bytes() == c.thd_workspace_layout(b, hq, b * s, b * s, True)[1]
            assert (api._thd_t_q, api._thd_t_kv, api._thd_head_pitch, api._thd_origins) == (b * s, b * s, s, c.DENSE_ORIGIN_SPECS)
        else:
            assert api.scratch_workspace_bytes() == dense_workspace_layout(b, hq, s, s)[1]


@pytest.mark.parametrize(
    "over, text",
    [
        pytest.param(dict(dtype=torch.float16), r"grouped d128 supports BF16 grouped attention only", id="fp16-grouped"),
        pytest.param(dict(hkv=4), r"cta_pair 'on' requires grouped heads", id="mha"),
        pytest.param(dict(s=320), r"grouped d128 dense sequence lengths must be multiples of 128", id="s-off-128"),
        pytest.param(dict(dkv_accum="fp32_reduce"), r"grouped d128 dkv_accum 'fp32_reduce' is served on the THD grouped chain only", id="fp32_reduce"),
        pytest.param(dict(pad_batch=True), r"grouped d128 q requires compact BSHD storage", id="padded-bshd"),
    ],
)
def test_cta_pair_dense_pair_declines_by_name(monkeypatch, over, text):
    """``on`` on a dense declaration the pair body cannot serve declines BY NAME through the REAL checks (a NotImplementedError naming the engine and
    the predicate, before any compile; never a silent one-CTA fallback): FP16 grouped (the partial storage), an MHA request (no pair body), a
    sequence off the 128-row dense tile, ``dkv_accum = "fp32_reduce"`` (the dense chains keep bf16 partials + the fp32 fold -- inherited from the
    one-CTA dense body, still declined on the pair: an open question, not a knob), a padded (non-compact) BSHD batch stride.  Under ``auto`` at the
    threshold the grouped rows decline the same way: the pair's decline falls back to the one-CTA dense body, which shares the grouped dense
    checks and declines too, the pair's decline recorded in the resolution's reason first."""
    c = _contract()
    fields = {k: v for k, v in over.items() if k in ("dkv_accum",)}
    geometry_kw = {k: v for k, v in over.items() if k in ("hkv", "s", "pad_batch")}
    api = _declaration_api(monkeypatch, _dense_geometry(**geometry_kw), dtype=over.get("dtype", torch.bfloat16), mode="on", **fields)
    with pytest.raises(NotImplementedError, match=f"{_ENGINE}: .*{text}"):
        api.check_support()
    assert not getattr(api, "_is_supported", False)
    if "hkv" in over:
        return  # MHA under auto / off takes the MHA routes (no heuristic there): pinned by test_cta_pair_on_requires_grouped_heads_typed
    threshold = c.cta_pair_min_seq((10, 0))  # the declaration api's part (``_declaration_api`` default cc)
    geometry_kw["s"] = threshold + 64 if "s" in over else threshold  # at the threshold (S % 128 == 64 keeps the off-tile row off the tile)
    api = _declaration_api(monkeypatch, _dense_geometry(**geometry_kw), dtype=over.get("dtype", torch.bfloat16), mode="auto", **fields)
    with pytest.raises(NotImplementedError, match=f"{_ENGINE}: .*{text}"):
        api.check_support()
    assert api._cta_pair.body == c.ONE_CTA_BODY and "pair body declined" in api._cta_pair.reason, api._cta_pair


def test_cta_pair_dense_graph_declaration_at_the_threshold_selects_the_pair(monkeypatch, compiler_version):
    """The graph API's plan-time hook, host-side (the engine row's declaration check: ``graph_analyzer.analyze`` -> ``engines.mismatch`` ->
    ``heuristics.recommend``, the eligibility suite's recipe) on a dense compact-BSHD BF16 GQA ``sdpa_backward`` graph at the 8K bar shape (B 4,
    Hq 32, Hkv 4, S = ``CTA_PAIR_MIN_SEQ[(10, 0)]``, causal): the d128 row (engine 20609) is recommended, and the shipping adapter built on the graph's declared
    geometry (what the engine row binds) resolves the pair body under the DEFAULT ``cta_pair`` ("auto") with the dense token -- the dispatch cell
    of the bar shape without the 8K oracle a device run would need.  The same graph at S 4096 resolves the one-CTA dense body."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.heuristics import recommend
    from test_sdpa_bwd_d128_thd_eligibility import reason
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    c = _contract()
    monkeypatch.setattr(ga, "_device_cc", lambda: (10, 0))
    for s, body, implementation in ((c.cta_pair_min_seq((10, 0)), c.PAIR_BODY, c.DENSE), (4096, c.ONE_CTA_BODY, _ONE_CTA_DENSE)):
        graph, _, _ = _half_bwd_graph(b=4, hq=32, hkv=4, d=128, thd=False, sq=s, skv=s, use_causal_mask=True)
        value = ga.analyze(graph)
        assert value is not None and not value.thd and (value.h_q, value.h_kv, value.s_q) == (32, 4, s) and reason(value) is None
        assert [plan.engine_id for plan in recommend("A", value, {_ENGINE: 20609})] == [20609]
        geometry = {name.lower(): (dim, stride) for name, dim, stride in value.port_layouts}
        geometry["stats"] = value.stats_t.get_dim(), value.stats_t.get_stride()
        api = _declaration_api(monkeypatch, geometry, causal=True)  # cta_pair left at the class default: "auto"
        assert type(api).cta_pair == "auto" and api.check_support() is True
        assert (api._cta_pair.body, api._cta_pair.token, api._implementation) == (body, f"pair=auto:{body}", implementation), (s, api._cta_pair)


# ------------------------------------------------------------------------------------------------------------ GPU rows (the CTA_PAIR_CCS parts)
def _require_pair_part():
    from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation

    _require_accumulation()


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7), (11, 0), (12, 0)])
def test_cta_pair_test_gates_match_forced_pair_support(monkeypatch, cc):
    """The suite's forced two-CTA gates admit exactly the devices the library admits (``CTA_PAIR_CCS``: cc 10.0 / 10.3 / 10.7 on this
    branch) and skip elsewhere with the forced-pair reason (#1985's alignment, read from the library's set rather than a copy)."""
    import d128_2cta_test_utils
    import test_sdpa_bwd_d128_dkv_accum_sm100 as dkv_tests
    from cudnn.frost import buffers

    c = _contract()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: cc)
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(dkv_tests, "make_adapter_2cta_stail", lambda case, **kwargs: case)
    _require_pair_part()
    for require_pair in (d128_2cta_test_utils._require_2cta, lambda: dkv_tests._make_pair(None)):
        if cc in c.CTA_PAIR_CCS:
            require_pair()
        else:
            with pytest.raises(pytest.skip.Exception, match="forced cta_group::2"):
                require_pair()


def _run(api, case, repeats=1):
    workspace = make_workspace(api.scratch_workspace_bytes())
    for _ in range(repeats):
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    return workspace


def _kernels(api, case, workspace):
    names = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            names.append(event.name)
    return [n for n in names if n.startswith("cudnn_kernel_")]


def test_cta_pair_on_declines_by_name_off_the_measured_part():
    """A REAL adapter on a CUDA device outside the admitted parts: ``on`` declines naming the cc predicate (observable off cc 10.0 / 10.3 / 10.7 only: the
    sm80 / sm120 CI lanes are this row's proof of the decline path)."""
    c = _contract()
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    cc = torch.cuda.get_device_capability()
    if cc in c.CTA_PAIR_CCS:
        pytest.skip(f"cc {cc} admits the pair body: the decline is observable off the SM100 line only")
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    api = make_adapter(case, compile_plan=False)
    api.cta_pair = "on"
    with pytest.raises(NotImplementedError, match=rf"{_ENGINE}: cta_pair 'on' requires cc .*got {re.escape(str(cc))}"):
        api.check_support()


@pytest.mark.parametrize(
    "mode, s, body, pair_map, layout",
    [
        pytest.param("auto", 256, "1cta", None, "thd", id="auto-s256-one-cta"),
        pytest.param("off", 8192, "1cta", None, "thd", id="off-s8k-one-cta"),
        pytest.param("auto", 8192, "2cta", "s_tail", "thd", id="auto-s8k-pair"),
        pytest.param("on", 256, "2cta", "s_tail", "thd", id="on-s256-pair"),
        pytest.param("auto", 256, "1cta", None, "dense", id="auto-s256-one-cta-dense"),
        pytest.param("auto", 8192, "2cta", "s_tail", "dense", id="auto-s8k-pair-dense"),
        pytest.param("on", 256, "2cta", "s_tail", "dense", id="on-s256-pair-dense"),
    ],
)
def test_cta_pair_launch_census_per_selection(mode, s, body, pair_map, layout):
    """Five launches under every selection (four on the one-CTA DENSE body, which has no setup launch); the MAIN kernel is the body the
    resolution names (the one-CTA grouped kernel of the layout below the threshold and under ``off``; the heuristic's pair body at S_max >=
    CTA_PAIR_MIN_SEQ under ``auto`` and at any S under ``on`` -- on the packed declaration AND on its dense compact-BSHD twin), the pair's
    cluster dims from its CFG; the suite's FP64 gate on every cell.  The 8K rows use a two-head single-sequence batch to keep the CPU
    reference short."""
    _require_pair_part()
    c = _contract()
    p = _prepared()
    cc = torch.cuda.get_device_capability()
    if body == "2cta" and cc not in c.CTA_PAIR_CCS:
        pytest.skip(f"cc {cc} is outside {c.CTA_PAIR_CCS}: the heuristic serves the one-CTA body here")
    hq, hkv = (2, 1) if s >= 8192 else (4, 2)
    case = make_case((s,), (s,), hq=hq, hkv=hkv, thd=layout == "thd", dtype=torch.bfloat16, causal=True)
    api = make_adapter(case, compile_plan=False)
    api.cta_pair = mode
    assert api.check_support()
    api.compile()
    res = p.cta_pair_resolution(api)
    assert res.body == body and res.token == f"pair={mode}:{body}", res
    workspace = _run(api, case)
    kernels = _kernels(api, case, workspace)
    if body == "1cta":
        assert len(kernels) == (5 if layout == "thd" else 4), kernels
        one_cta = _ONE_CTA_KERNEL if layout == "thd" else _ONE_CTA_DENSE_KERNEL
        assert any(one_cta in n for n in kernels) and not any("2cta" in n for n in kernels), kernels
        assert api._prepared.artifact.implementation == _ONE_CTA_IMPLEMENTATIONS[layout]
    else:
        assert len(kernels) == 5, kernels
        assert any(_PAIR_KERNEL in n for n in kernels) and any("setup_thd_d128_2cta_kernel" in n for n in kernels), kernels
        assert api._prepared.artifact.implementation == c.IMPLEMENTATIONS[layout] and res.pair_map == pair_map
        import importlib

        cfg = importlib.import_module("cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16").CFG  # the pair body's config binding (section 8)
        assert (cfg.CGA_M, cfg.CGA_N) == (c.CTA_PAIR_CLUSTER, 1)


@pytest.mark.parametrize(
    "lens_q, lens_kv, hq, hkv, causal",
    [
        pytest.param((256, 256), (256, 256), 4, 2, True, id="uniform-256-causal"),
        pytest.param((384, 384), (384, 384), 4, 2, False, id="uniform-384-noncausal"),  # S % 128 == 0, S % 256 != 0: the half-dead last pair
        pytest.param((1024,), (1024,), 8, 1, True, id="uniform-1024-mqa8-causal"),
        # RECTANGULAR (S_q != S_kv): the only rows that tell the setup kernel's four dense writes apart (kv_lens[b] = S_kv, cu_q[b] = b * S_q,
        # cu_k[b] = b * S_kv, org[8][b] = b * H * S_q -- a copy with dense_sq / dense_skv swapped in any of them renders every SQUARE row identical
        # and reads sequence 1's Q / K / Stats from the wrong token here).  S_q < S_kv under top-left causal = keys past S_q - 1 never attended
        # (zero dK / dV rows); S_q > S_kv unmasked = the q rows past S_kv - 1 see every key.  RED-first: the swap in a scratch copy of the helper.
        pytest.param((256, 256), (512, 512), 4, 2, True, id="rect-sq256-skv512-causal"),
        pytest.param((512, 512), (256, 256), 4, 2, False, id="rect-sq512-skv256-noncausal"),
    ],
)
def test_cta_pair_dense_twin_matches_thd_uniform(lens_q, lens_kv, hq, hkv, causal):
    """The dense twin IS the THD chain on a uniform batch: the same per-sequence data declared once as dense compact BSHD and once as packed THD
    (compact, no gaps) under ``cta_pair = "on"`` -- both pass the suite's FP64 gate (``check_case``: every gradient, the storage guards, the
    workspace poison), and dK / dV are BITWISE equal between the two forms (the same kernels, the same bf16 partials folded in the same order;
    ``torch.equal`` on the int16 views, never a tolerance); dQ (fp32 reduce-add, unordered) is held by the gate on each side.  The dense plan
    carries the dense implementation token, nine native roles and no length operands.  The two RECTANGULAR rows (S_q != S_kv, uniform per
    side) are the detector of the dense arm's extents: on a square batch the synthesized metadata is the same whichever extent each write reads."""
    from d128_grouped_test_utils import GRADS

    _require_pair_part()
    c = _contract()
    cc = torch.cuda.get_device_capability()
    if cc not in c.CTA_PAIR_CCS:
        pytest.skip(f"cc {cc} is outside {c.CTA_PAIR_CCS}: the pair body is not admitted here")
    lens = (lens_q, lens_kv)
    thd_case = make_case(lens_q, lens_kv, hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=causal, gaps=False, origin_ports=())
    dense_case = make_case(lens_q, lens_kv, hq=hq, hkv=hkv, thd=False, dtype=torch.bfloat16, causal=causal, sequences=thd_case.sequences)
    outputs = {}
    for case in (thd_case, dense_case):
        api = make_adapter(case, compile_plan=False)
        api.cta_pair = "on"
        assert api.check_support() and api._cta_pair.body == c.PAIR_BODY
        api.compile()
        assert api._prepared.artifact.implementation == c.IMPLEMENTATIONS["thd" if case.thd else "dense"]
        if not case.thd:
            assert not api._prepared.length_form and len(api._prepared.operands) == 9 and api._prepared.roles == tuple(api._prepared.roles[:9])
        _run(api, case, repeats=2)
        outputs[case.thd] = {role: case.tensors[role][0].clone() for role in GRADS}
    for role in ("dk", "dv"):
        a, b = outputs[True][role], outputs[False][role]
        assert a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{role}: the dense twin is not bitwise the THD chain"
    # dQ is a reduce-add output (fp32 TMA reduce-add in per-launch order, > 2 contributions per element past one kv pair): a TOLERANCE twin held
    # by the fp64 gate on each side; its distance between the two forms is printed as a NOTE (magnitude, the fraction of bitwise-equal elements),
    # never asserted bitwise (G09: a bitwise claim is by construction).
    a, b = outputs[True]["dq"].float(), outputs[False]["dq"].float()
    diff = (a - b).abs()
    i = int(diff.argmax())
    print(
        f"\nNOTE dQ dense vs THD-uniform ({lens}, hq {hq}, hkv {hkv}, causal {causal}): max |diff| {diff.max().item():.3e} at flat {i} "
        f"(|dQ| {a.flatten()[i].abs().item():.3e}), mean |diff| {diff.mean().item():.3e}, bitwise-equal elements {(diff == 0).float().mean().item():.4f}"
    )


def test_cta_pair_dense_twin_on_tmem_map_dq576_matches_s_tail():
    """The dense twin x ``tmem_map = "dq_576"``: the one declaration that reaches BOTH sides of the merged host files at once (``layout_form_of(False)``
    x ``resolve_tmem_map("dq_576", (10, 7))`` in ``compile_host``'s key / signature and ``compile_plan``'s ``layout=`` + ``tmem_map=``); it had never
    been launched before this row (reviewer, the merge).  On the part that serves the map (cc 10.7 -- ``TMEM_MAP_PARTS``, the library's own set;
    skips BY NAME elsewhere) a dense compact-BSHD GQA declaration under ``cta_pair = "on"`` runs on EACH map through the suite's fp64 gate (two
    executes on a poisoned workspace), the plan records the dense token and the resolver the map by name, and dK / dV are BITWISE between the maps on
    the same data (the map moves the dQ accumulator's TMEM columns and deletes two barrier rows: no GEMM, k-order or rounding change -- a claim by
    construction, as the THD twin ``test_2cta_stail_tmem_maps_agree_bitwise`` pins); dQ bitwise too on this cell (one kv pair per sequence: at most
    ``_DQ_ORDER_FREE_PAIRS`` reduce-add contributions per element)."""
    from test_sdpa_bwd_d128_2cta_stail_sm100 import _DQ_ORDER_FREE_PAIRS, _TMEM_MAPS, _require_tmem_map

    _require_pair_part()
    _require_tmem_map("dq_576")
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    c = _contract()
    p = _prepared()
    lens, hq, hkv = (256, 256), 4, 2
    case = make_case(lens, lens, hq=hq, hkv=hkv, thd=False, dtype=torch.bfloat16, causal=True)
    assert max(lens) <= _DQ_ORDER_FREE_PAIRS * CFG.PAIR_KV_ROWS  # the dQ claim below is by construction: name the cell
    grads = {}
    for name in _TMEM_MAPS:
        api = make_adapter(case, compile_plan=False)
        api.cta_pair = "on"
        api.tmem_map = name
        assert api.check_support() and api._cta_pair.body == c.PAIR_BODY and p.tmem_map(api) == name
        api.compile()
        assert api._prepared.artifact.implementation == c.DENSE and not api._prepared.length_form
        _run(api, case, repeats=2)
        grads[name] = {role: case.tensors[role].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dk", "dv", "dq"):
        assert torch.equal(
            grads["s_tail"][role].view(torch.int16), grads["dq_576"][role].view(torch.int16)
        ), f"{role}: dense x s_tail vs dense x dq_576 differ bitwise"


def test_cta_pair_dense_twin_capture_replay_and_rebind():
    """RED-first: the dense twin under CUDA-graph capture (the training step's form) -- eager x2 on a poisoned workspace (0xFF = NaN in every partial
    dtype: a partial never written fails the gate) and canaried storage, one capture, replays x3 each re-poisoned, the suite's fp64 gate on every
    gradient plus the storage guards and the workspace prefix / suffix each time; then the SAME storage refilled with another dataset
    (``copy_case_storage``: data and reference change, no captured address does) and replayed -- the captured launch reads the metadata the setup
    kernel synthesizes from the declaration, nothing host-side.  dK / dV are bitwise between the eager run and every replay of the first dataset
    (one launch, one dataset, the fold in a fixed order)."""
    from d128_grouped_test_utils import copy_case_storage

    _require_pair_part()
    c = _contract()
    cc = torch.cuda.get_device_capability()
    if cc not in c.CTA_PAIR_CCS:
        pytest.skip(f"cc {cc} is outside {c.CTA_PAIR_CCS}: the pair body is not admitted here")
    lens, hq, hkv = (256, 256), 4, 2
    case = make_case(lens, lens, hq=hq, hkv=hkv, thd=False, dtype=torch.bfloat16, causal=True)
    api = make_adapter(case, compile_plan=False)
    api.cta_pair = "on"
    assert api.check_support() and api._cta_pair.body == c.PAIR_BODY
    api.compile()
    assert api._prepared.artifact.implementation == c.DENSE
    workspace = _run(api, case, repeats=2)
    eager = {role: case.tensors[role].clone() for role in ("dk", "dv")}
    args = adapter_arguments(case)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        api.execute(**args, workspace=workspace.tensor)
    torch.cuda.current_stream().wait_stream(stream)
    try:
        for _ in range(3):
            poison(case, workspace)
            graph.replay()
            torch.cuda.synchronize()
            check_case(case, workspace)
            for role in ("dk", "dv"):
                assert torch.equal(case.tensors[role].view(torch.int16), eager[role].view(torch.int16)), f"{role}: replay is not bitwise the eager launch"
        changed = make_case(lens, lens, hq=hq, hkv=hkv, thd=False, dtype=torch.bfloat16, causal=True, seed=151)
        copy_case_storage(case, changed)
        for _ in range(2):
            poison(case, workspace)
            graph.replay()
            torch.cuda.synchronize()
            check_case(case, workspace)
    finally:
        graph.reset()
