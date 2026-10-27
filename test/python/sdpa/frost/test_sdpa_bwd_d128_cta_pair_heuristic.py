# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-S heuristic that routes the BF16 grouped THD d128 backward between the one-CTA body and the cta_group::2 pair body.

The typed adapter field ``SdpaBwdDslSm100D128.cta_pair`` ("auto" = the default: the per-S rule | "off" = the one-CTA body always |
"on" = the pair body, declining BY NAME when a predicate fails) is resolved by ``prepared_sm100_d128_grouped_2cta`` from ONE set of
config constants in ``d128_grouped_2cta_contract`` (``CTA_PAIR_MIN_SEQ`` = the smallest S where the pair body won the in-session
step on cc 10.0; ``CTA_PAIR_CCS`` = the measured part; the cluster-of-two admission through the DSL's occupancy query) plus every
support predicate of the pair body.  The host rows need no GPU: the pure selector, the typed domain, the declines by name, the
dispatch with its fallback, the resolution token.  The GPU rows (skipped off SM100 / SM103) pin the launch census per selection:
five launches either way, the pair's cluster dims, the one-CTA kernel below the threshold and under ``off``.

RED-first: every row below was written before the field existed.
"""

import re
from types import SimpleNamespace

import pytest
import torch

from d128_grouped_test_utils import adapter_arguments, check_case, make_adapter, make_case, make_workspace, poison
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm100_d128"
_ONE_CTA_KERNEL = "bprop_d128_accum_grouped_thd_kernel"
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
    kw = dict(thd=True, s_q_max=max(lens), s_kv_max=max(lens), cc=(10, 0), max_active_pairs=76)
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

    assert c.CTA_PAIR_MODES == ("auto", "off", "on") and c.CTA_PAIR_MODE_DEFAULT == "auto"
    assert c.CTA_PAIR_MIN_SEQ == 8192 and c.CTA_PAIR_MIN_SEQ % c.UNIT_ROWS == 0 and c.CTA_PAIR_MIN_SEQ % CFG.PAIR_KV_ROWS == 0
    assert c.CTA_PAIR_CCS == ((10, 0),) and all(tuple(cc) in c.ACCUMULATION_CCS for cc in c.CTA_PAIR_CCS)
    assert c.CTA_PAIR_CLUSTER == CFG.CGA_M == 2
    assert c.CTA_PAIR_BODY == "s_tail" == STAIL_CFG.TMEM_MAP
    assert (c.PAIR_BODY, c.ONE_CTA_BODY) == ("2cta", "1cta")
    assert "smallest S where the pair body won" in c.__doc__ or "smallest S where the pair body won" in open(c.__file__).read()


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
        assert res.cluster_admitted is True and str(c.CTA_PAIR_MIN_SEQ) in res.reason
    else:
        assert "seq" in res.reason and str(c.CTA_PAIR_MIN_SEQ) in res.reason


def test_cta_pair_auto_reads_both_q_and_kv_extents():
    """A lopsided batch (one side below the threshold) falls to the measured-safe one-CTA body: the rule is the conjunction."""
    c = _contract()
    assert _select("auto", 16384, "uniform", s_kv_max=4096).body == c.ONE_CTA_BODY
    assert _select("auto", 16384, "uniform", s_q_max=4096).body == c.ONE_CTA_BODY
    assert _select("auto", 16384, "uniform", s_q_max=8192, s_kv_max=16384).body == c.PAIR_BODY


def test_cta_pair_auto_falls_back_by_named_predicate():
    """Each predicate the pair needs sends ``auto`` to the one-CTA body with the predicate NAMED in the reason: THD (no dense pair body), the
    measured part (cc 10.0: SM103 is admitted by the body but unmeasured, so it serves the one-CTA body until measured), the cluster-of-two
    admission (0 active pairs from the occupancy query), a device that is not CUDA at all."""
    c = _contract()
    for over, name in (
        (dict(thd=False), "thd"),
        (dict(cc=(10, 3)), "cc"),
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
    assert _select("auto", 16384, "uniform", cc=(10, 3), max_active_pairs=probe).body == c.ONE_CTA_BODY and calls == []
    assert _select("auto", 16384, "uniform", max_active_pairs=probe).body == c.PAIR_BODY and calls == [1]


def test_cta_pair_on_declines_by_name_and_ignores_the_s_rule():
    """``on`` forces the pair body at ANY S (the threshold is the heuristic's, not a support predicate) and declines BY NAME when a pair predicate
    fails: the decline text names the predicate (thd / cc / cluster) and the value seen."""
    c = _contract()
    for s_max in (256, 2048, 8192, 32768):
        res = _select("on", s_max, "uniform")
        assert res.body == c.PAIR_BODY and res.token == "pair=on:2cta", res
    with pytest.raises(ValueError, match=r"cta_pair 'on' requires THD"):
        _select("on", 16384, "uniform", thd=False)
    for cc in ((10, 3), (9, 0), (12, 0)):
        with pytest.raises(ValueError, match=rf"cta_pair 'on' requires cc .*\(10, 0\).*got {re.escape(str(cc))}"):
            _select("on", 16384, "uniform", cc=cc)
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
    c = _contract()
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
    """The decline texts and the reasons are stable tokens a driver pins by value (R4): the predicate names are the three the tracker row lists."""
    c = _contract()
    assert c.CTA_PAIR_PREDICATES == ("thd", "cc", "cluster", "seq")
    res = _select("auto", 2048, "uniform")
    assert res.reason.startswith("cta_pair=auto: the one-CTA body (seq:")
    res = _select("auto", 8192, "uniform")
    assert res.reason.startswith("cta_pair=auto: the pair body (")


# ------------------------------------------------------------------------------------------------------------ GPU rows (SM100 / SM103)
def _require_pair_part():
    from d128_2cta_test_utils import _require_2cta

    _require_2cta()


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
    """A REAL adapter on a CUDA device that is not the measured part: ``on`` declines naming the cc predicate (observable only off cc 10.0)."""
    c = _contract()
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    cc = torch.cuda.get_device_capability()
    if cc in c.CTA_PAIR_CCS:
        pytest.skip(f"cc {cc} is the measured part: the decline is observable off it only")
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    api = make_adapter(case, compile_plan=False)
    api.cta_pair = "on"
    with pytest.raises(NotImplementedError, match=rf"{_ENGINE}: cta_pair 'on' requires cc .*got {re.escape(str(cc))}"):
        api.check_support()


@pytest.mark.parametrize(
    "mode, s, body, pair_map",
    [
        pytest.param("auto", 256, "1cta", None, id="auto-s256-one-cta"),
        pytest.param("off", 8192, "1cta", None, id="off-s8k-one-cta"),
        pytest.param("auto", 8192, "2cta", "s_tail", id="auto-s8k-pair"),
        pytest.param("on", 256, "2cta", "s_tail", id="on-s256-pair"),
    ],
)
def test_cta_pair_launch_census_per_selection(mode, s, body, pair_map):
    """Five launches under every selection; the MAIN kernel is the body the resolution names (the one-CTA grouped THD kernel below the threshold
    and under ``off``; the heuristic's pair body at S_max >= CTA_PAIR_MIN_SEQ under ``auto`` and at any S under ``on``), the pair's cluster dims
    from its CFG; the suite's FP64 gate on every cell.  The 8K rows use a two-head single-sequence batch to keep the CPU reference short."""
    _require_pair_part()
    c = _contract()
    p = _prepared()
    cc = torch.cuda.get_device_capability()
    if body == "2cta" and cc not in c.CTA_PAIR_CCS:
        pytest.skip(f"cc {cc} is not the measured part: the heuristic serves the one-CTA body here")
    hq, hkv = (2, 1) if s >= 8192 else (4, 2)
    case = make_case((s,), (s,), hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=True)
    api = make_adapter(case, compile_plan=False)
    api.cta_pair = mode
    assert api.check_support()
    api.compile()
    res = p.cta_pair_resolution(api)
    assert res.body == body and res.token == f"pair={mode}:{body}", res
    workspace = _run(api, case)
    kernels = _kernels(api, case, workspace)
    assert len(kernels) == 5, kernels
    if body == "1cta":
        assert any(_ONE_CTA_KERNEL in n for n in kernels) and not any("2cta" in n for n in kernels), kernels
        assert api._prepared.artifact.implementation == "accumulation_grouped_thd"
    else:
        assert any(_PAIR_KERNEL in n for n in kernels) and any("setup_thd_d128_2cta_kernel" in n for n in kernels), kernels
        assert api._prepared.artifact.implementation == c.THD and res.pair_map == pair_map
        import importlib

        cfg = importlib.import_module("cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_grouped_2cta_stail_f16").CFG
        assert (cfg.CGA_M, cfg.CGA_N) == (c.CTA_PAIR_CLUSTER, 1)
