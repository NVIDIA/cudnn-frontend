# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The third softmax warp group of the Rubin f16/bf16 d128-file body (``TemplateParams.softmax_wgs = 3``): the MEASURED default of ONE class
(``config_sm107.sm107_softmax_wgs_pick``: bias + kv_mask on the pair item at TILE_N 64, cga1, dense KV lengths, ``s_kv >= 640``, ``b_flat
<= 3072``) and a public ``softmax_wgs=3`` request elsewhere on the pair item.

The (n, n+1) pair work item (``n_chunk >= 1``) at TILE_N 64 can run THREE softmax warp groups: a 20-warp body (WG0 / WG1 / WG2 at warps
0-11, correction 12-15, MMA 16, TMA-LDG 17, TMA-STG 18, scheduler 19) whose step takes three rows of the chunk, one ring stage per
sub-tile per kv tile, three Q / O slabs, three stages of every per-sub-tile ring, the register split 12 x 128 + 4 x 56 + 4 x 40 = 1920 =
the 20-warp entry pool (12 x 120 + 4 x 80 + 4 x 40 at the 64-wide tile), the P-publish hand-off a 3-way chain over named barriers 8 and 9.
Every value computes the SAME function: a row's softmax runs the same instruction stream whichever group owns it, so O and Stats are
BITWISE the two-group kernel's.  The arm is served on the pair item at tile_n 64 / cga 1 on the f16/bf16 32- / 64-wide tiles with a bias
and declined typed everywhere else (the one-batch-entry paths hand-unroll two sub-tiles; a 128-register softmax holds one 64-column chunk).

Tiers: CPU contract (the record field, the measured rule, the factory's derived warp table / split / arriver counts, the pool equality, the
TMEM map, the bias budget at three slabs, the typed declines of the factory and the adapter incl. the knob-less rule cells), host rendering pins
(sm_107a trace-compile: the three-group renderings of the default class carry the split (``USETMAXREG`` 4 = 2 INCREASE sites + 2 DECREASE), no
cluster drain, the bit-word mask form, and spill within the measured bound; the two-group renderings' byte identity to the committed records is
pinned by the streamed module), GPU (Rubin): the DEFAULT class knob-less on both tiles (the pick -> record -> plan chain end to end, the
record's ``softmax_wgs`` read off the loaded module) and the public option on the bias-only pair item (a ring wrap with a dead chunk tail, a
zero-length row in a chunk) -- the fp64 oracle and BITWISE equality with the two-group kernel.
"""

import dataclasses

import pytest
import torch

from frost_test_utils import assert_no_new_spills, requires_rubin, run_sass_probe, sass_probe_source
import test_sdpa_bias_sm107 as _s13

pytestmark = [pytest.mark.L0]


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


@pytest.fixture
def _pins(monkeypatch):
    """Pin the softmax warp-group count, the chunk length, the bias form and the ring depth a knob-less plan records (the config module's
    private, process-local seams); ``None`` = the pick."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(softmax_wgs=None, n_chunk=None, mode=None, stages=None):
        monkeypatch.setattr(c, "_SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS", softmax_wgs)
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", n_chunk)
        monkeypatch.setattr(c, "_BIAS_MODE_OVERRIDE_FOR_SWEEPS", mode)
        monkeypatch.setattr(c, "_BIAS_STAGES_OVERRIDE_FOR_SWEEPS", stages)

    return _set


def _params(d=32, **kw):
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    return TemplateParams(**{**dict(cta_mma=1, dtype_qkv=2, d_flavor=d, bias_mode=1, n_chunk=4, tile_n=64), **kw})


def _make(d):
    from cudnn.sdpa.fwd import config_sm107 as c

    return c.make_cfg_d32 if d == 32 else c.make_cfg_d64


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_the_record_field_and_the_seam_are_appended_and_default_to_two():
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    # `s_slots` (the double-buffered S of the small tiles) is appended after it: the tail names both
    names = [f.name for f in dataclasses.fields(TemplateParams)]
    names = names[: names.index("q_in_tmem") + 1]  # Later appended fields are legal.
    assert names[-3:] == ["softmax_wgs", "s_slots", "q_in_tmem"] and TemplateParams().softmax_wgs == 2
    assert c._SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS is None, "the seam is a per-test / per-sweep pin, never a module default"
    assert c.SMALL_D_SOFTMAX_WGS == frozenset({2, 3}) and "SMALL_D_SOFTMAX_WGS" in c.__all__ and "sm107_softmax_wgs_pick" in c.__all__
    assert c.sm107_softmax_wgs_pick() == 2 and c.sm107_softmax_wgs_pick(3) == 3


def test_the_measured_default_rule_and_the_triple_step_chunk():
    """The MEASURED default: 3 groups for a bias + kv_mask graph on the pair item at tile_n 64 / cga 1 with dense KV lengths from
    S_kv 640 (D 32 S 640 +8.1 %, S 1024 +2.8 %; D 64 S 768 +3.5 % over the pair's pick in one process each) -- 2 everywhere else (the bias alone
    at TILE_N 64 loses 1.0..11.6 %; S 384 / 512 sit inside the protocol's resolution).  The chunk on the 3-WG body is a multiple of THREE:
    the pair's cap 8 -> 6 (n9 measured -2.7..-3.6 % against n6), the one-tile floor 4 -> 3; every pair pick (rows_per_step 2) is unchanged."""
    from cudnn.sdpa.fwd import config_sm107 as c

    P = c.sm107_softmax_wgs_pick
    assert c.SM107_THREE_WG_S_KV_MIN == 640 and "SM107_THREE_WG_S_KV_MIN" in c.__all__
    assert c.SM107_THREE_WG_B_FLAT_MAX == 3072 and "SM107_THREE_WG_B_FLAT_MAX" in c.__all__
    base = dict(d_flavor=32, s_kv=1024, tile_n=64, pair=True, has_bias=True, has_kv_mask=True, b_flat=1024)
    assert (
        P(**{**base, "b_flat": 3072}) == 3 and P(**{**base, "b_flat": 3073}) == 2 and P(**{**base, "b_flat": None}) == 2
    ), "the flattened-batch gate (4096 rows = the S 1024 B 4 cell lost 2.4 %; 3072 rows = S 768 B 4 won)"
    assert P(**base) == 3 and P(**{**base, "d_flavor": 64, "s_kv": 768}) == 3 and P(**{**base, "s_kv": 640}) == 3
    assert P(**{**base, "s_kv": 512}) == 2 and P(**{**base, "s_kv": 384}) == 2
    assert P(**{**base, "has_kv_mask": False}) == 2 and P(**{**base, "tile_n": 128}) == 2 and P(**{**base, "pair": False}) == 2
    assert P(**{**base, "padded": True}) == 2 and P(**{**base, "cga": 2}) == 2 and P(**{**base, "d_flavor": 128}) == 2 and P(**{**base, "has_bias": False}) == 2
    assert P(2, **base) == 2 and P(3, **{**base, "s_kv": 128}) == 3, "a request wins over the rule (the adapter validates it)"
    N = c.sm107_n_chunk_pick
    for s, d in ((1024, 32), (640, 32), (768, 64), (512, 32)):
        assert N(1, s, 4, s, 212, 1, s_kv=s, d_flavor=d, tile_n=64) == 8 and N(1, s, 4, s, 212, 1, s_kv=s, d_flavor=d, tile_n=64, rows_per_step=3) == 6, (s, d)
    assert (
        N(1, 384, 4, 384, 212, 1, s_kv=384, d_flavor=32, tile_n=64) == 4 and N(1, 384, 4, 384, 212, 1, s_kv=384, d_flavor=32, tile_n=64, rows_per_step=3) == 3
    )
    assert c.sm107_n_chunk_rule(400, 4, 400, 212, rows_per_step=3) % 3 == 0 and c.sm107_n_chunk_refine(3, 400, 4, 400, 212, s_kv=400, rows_per_step=3) == 3
    assert c.sm107_n_chunk_refine(2, 6, 4, 128, 212, s_kv=128) == 4 and c.sm107_n_chunk_refine(2, 6, 4, 128, 212, s_kv=128, rows_per_step=3) == 3
    assert c.sm107_n_chunk_rule(1, 4, 128, 212, rows_per_step=3) == 0, "a single row keeps the one-batch-entry item"


@pytest.mark.parametrize("d", [32, 64])
def test_the_factory_derives_the_twenty_warp_table_the_split_and_the_arriver_counts(d):
    """The three-group body as the config: 20 warps, one group per sub-tile, the role ids contiguous, READ_TILE_ARRIVERS 19 (12 softmax + 4
    correction + MMA + TMA-LDG + TMA-STG; the scheduler warp waits it), BIAS_EMPTY_ARRIVERS 12 (one elected arrive per softmax warp), the split EQUAL to the
    launch's entry pool at both counts (frost-kernels.md s2: `SUM(role regs x warps) == entry x warps`)."""
    from cudnn.sdpa.bwd.config_sm107 import reg_entry_pool

    for wgs in (2, 3):
        cfg, _ = _make(d)(_params(d, softmax_wgs=wgs))
        assert (cfg.SOFTMAX_WARPGROUPS, cfg.TILES_Q, cfg.TOTAL_WARPS, cfg.THREADS_PER_CTA) == (wgs, wgs, 8 + 4 * wgs, 32 * (8 + 4 * wgs))
        assert (cfg.SOFTMAX_WG0_BASE, cfg.SOFTMAX_WG1_BASE, cfg.CORR_WARP_BASE) == (0, 4, 4 * wgs)
        assert (cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == tuple(4 * wgs + 4 + i for i in range(4))
        assert cfg.READ_TILE_ARRIVERS == 4 * wgs + 4 + 3 and cfg.BIAS_EMPTY_ARRIVERS == 4 * wgs
        split = wgs * 4 * cfg.SOFTMAX_REGS + 4 * cfg.CORRECTION_REGS + 4 * cfg.OTHER_REGS
        assert split == reg_entry_pool(cfg.TOTAL_WARPS), (wgs, d, split)
        assert (cfg.SOFTMAX_REGS, cfg.CORRECTION_REGS) == {(2, 32): (192, 88), (2, 64): (192, 88), (3, 32): (128, 56), (3, 64): (120, 80)}[(wgs, d)]
        assert cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS == 40
    assert reg_entry_pool(16) == 2048 and reg_entry_pool(20) == 1920


def test_the_tmem_map_and_the_bias_budget_at_three_slabs():
    """TMEM: S 0 / 64 / 128 (P in each tail at +32), O 192 / 224 / 256 (ends 288 <= STATS 544) at D 32; 192 / 256 / 320 (ends 384) at D 64.
    SMEM: the third Q and O slab cost one resident bias tile at the 32-wide tile (15 -> 14 tiles: S_kv <= 896 resident) and one at the
    64-wide one (11 -> 9: 704 -> 576, two tiles: the 64-wide slabs are twice the size), the SAME arithmetic the adapter's budget reads (`tiles_q`)."""
    from cudnn.sdpa.fwd import config_sm107 as c

    cfg32, _ = _make(32)(_params(32, softmax_wgs=3))
    lay = c.d128_family_tmem_layout(cfg32)
    assert (lay.S_OFF, lay.P_OFF, lay.O_OFF, lay.STATS_OFF) == ((0, 64, 128), (32, 96, 160), (192, 224, 256), 544)
    cfg64, _ = _make(64)(_params(64, softmax_wgs=3))
    lay64 = c.d128_family_tmem_layout(cfg64)
    assert lay64.O_OFF == (192, 256, 320) and lay64.O_OFF[-1] + 64 <= lay64.STATS_OFF
    assert (c.sm107_bias_resident_s_kv_max(32, 1, tile_n=64, pair=True), c.sm107_bias_resident_s_kv_max(32, 1, tile_n=64, pair=True, tiles_q=3)) == (960, 896)
    assert (c.sm107_bias_resident_s_kv_max(64, 1, tile_n=64, pair=True), c.sm107_bias_resident_s_kv_max(64, 1, tile_n=64, pair=True, tiles_q=3)) == (704, 576)
    assert cfg32.BIAS_RESIDENT_TILES == 14 and cfg64.BIAS_RESIDENT_TILES == 9
    # the streamed ring keeps its deepest depth beside the three slabs
    assert c.sm107_bias_stages_pick(32, 1, tile_n=64, pair=True, tiles_q=3) == 3


# the factory's own declines; the TILE_N 128 / one-batch-entry / no-bias cells are the adapter twin's below (the same predicate, typed there)
@pytest.mark.parametrize(
    "kw, match",
    [(dict(cta_mma=2, d_flavor=64), "cga1"), (dict(softmax_wgs=4), "softmax_wgs must be one of"), (dict(softmax_wgs=1), "softmax_wgs must be one of")],
    ids=["cga2", "four_groups", "one_group"],
)
def test_the_factory_declines_the_third_group_off_its_serving_set(kw, match):
    from cudnn.sdpa.fwd import config_sm107 as c

    d = kw.pop("d_flavor", 32)
    with pytest.raises(ValueError, match=match):
        _make(d)(_params(d, **{**dict(softmax_wgs=3), **kw}))
    # the validator's own predicates (sdpa-invariants s6): a config whose body disagrees with its warp table raises, by name
    cfg, _ = _make(32)(_params(32, softmax_wgs=3))
    with pytest.raises(ValueError, match="READ_TILE_ARRIVERS"):
        c._validate_softmax_wgs(dataclasses.replace(cfg, READ_TILE_ARRIVERS=15), "probe")
    with pytest.raises(ValueError, match="entry pool"):
        c._validate_softmax_wgs(dataclasses.replace(cfg, SOFTMAX_REGS=136), "probe")
    with pytest.raises(ValueError, match="one softmax warp group per Q sub-tile"):
        c._validate_softmax_wgs(dataclasses.replace(cfg, TILES_Q=2), "probe")
    with pytest.raises(ValueError, match="role table"):
        c._validate_softmax_wgs(dataclasses.replace(cfg, CORR_WARP_BASE=8), "probe")


@pytest.fixture
def _rubin_adapter(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a Rubin device on any host (the bias module's fixture)."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


@pytest.mark.parametrize(
    "kw, accept",
    [
        (dict(s_q=384, s_kv=384, b=4, bias=True, tile_n=64, softmax_wgs=3), True),
        (dict(s_q=384, s_kv=384, b=4, bias=True, tile_n=128, softmax_wgs=3), False),
        (dict(s_q=384, s_kv=384, b=4, bias=True, tile_n=64, n_chunk=0, softmax_wgs=3), False),
        (dict(s_q=384, s_kv=384, b=4, bias=False, tile_n=64, softmax_wgs=3), False),
        (dict(s_q=384, s_kv=384, b=4, bias=True, tile_n=64, softmax_wgs=3, d=64), True),
        (dict(s_q=384, s_kv=384, b=4, bias=True, tile_n=64, softmax_wgs=2), True),
        (dict(s_q=1024, s_kv=1024, b=1024, bias=True, kv_mask=True), True),
        (dict(s_q=512, s_kv=512, b=512, bias=True, kv_mask=True), True),
        (dict(s_q=1024, s_kv=1024, b=8, bias=True), True),
    ],
    ids=[
        "pair_t64_wg3",
        "t128_declines",
        "one_batch_entry_declines",
        "no_bias_declines",
        "d64_pair_t64_wg3",
        "wg2_default",
        "knobless_both_1024_rule_3",
        "knobless_both_512_rule_2",
        "knobless_bias_1024_rule_2",
    ],
)
def test_adapter_check_support_is_the_factorys_twin(_rubin_adapter, kw, accept):
    """The standalone adapter honours ``softmax_wgs=3`` on the pair item at tile_n 64 / cga 1 and declines it typed elsewhere (rule 8b: the
    adapter and the factory read the same predicate); the record it lowers carries the count."""
    kw = dict(kw)
    d = kw.pop("d", 32)
    bias = kw.pop("bias")
    kv_mask = kw.pop("kv_mask", False)
    s_q, s_kv, b = kw.pop("s_q"), kw.pop("s_kv"), kw.pop("b")
    api = _s13._api(d, s_q=s_q, s_kv=s_kv, b=b, h=4, bias=(1, 4, s_q, s_kv) if bias else None, kv_mask=(b, 1, 1, s_kv) if kv_mask else None, **kw)
    if accept:
        assert api.check_support() is True
        params = api.template_params()
        if "softmax_wgs" in kw:
            assert params.softmax_wgs == kw["softmax_wgs"] and params.tile_n == 64 and params.n_chunk >= 1
        else:
            # knob-less: the measured rule -- 3 groups on the bias + kv_mask pair form at tile_n 64 from S_kv 640, else 2 -- and the chunk a
            # multiple of the body's rows per step at THIS host's SM count and the plan's flattened batch (the config's own arithmetic, rule 8b')
            from cudnn.sdpa.fwd import api_dsl as _api
            from cudnn.sdpa.fwd import config_sm107 as c

            want = 3 if (kv_mask and s_kv >= c.SM107_THREE_WG_S_KV_MIN and b <= c.SM107_THREE_WG_B_FLAT_MAX) else 2
            assert params.softmax_wgs == want, (kw, params)
            assert params.tile_n == (64 if kv_mask else 128), params
            if kv_mask:
                sms = _api._device_sm_count(torch.device("cuda"))
                want_n = c.sm107_n_chunk_pick(1, b, 4, s_q, sms, 1, s_kv=s_kv, d_flavor=d, tile_n=64, rows_per_step=want)
                assert params.n_chunk == want_n and params.n_chunk % want == 0 and params.n_chunk >= want, (params.n_chunk, want_n)
            else:
                assert params.n_chunk == 0, params  # the bias alone at S 1024: the one-batch-entry streamed ring at TILE_N 128 (the pick)
    else:
        with pytest.raises((NotImplementedError, ValueError), match="softmax_wgs"):
            api.check_support()


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("SOFTMAX_WARPGROUPS", mod.CFG.SOFTMAX_WARPGROUPS)
print("TOTAL_WARPS", mod.CFG.TOTAL_WARPS)
print("TILES_Q", mod.CFG.TILES_Q)
print("SOFTMAX_REGS", mod.CFG.SOFTMAX_REGS)
print("CORRECTION_REGS", mod.CFG.CORRECTION_REGS)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("BIAS_RESIDENT_TILES", mod.CFG.BIAS_RESIDENT_TILES)
print("STAGES_BIAS", mod.CFG.STAGES_BIAS)
print("READ_TILE_ARRIVERS", mod.CFG.READ_TILE_ARRIVERS)
print("BIAS_PRELOAD_EFF", mod._BIAS_PRELOAD_EFF)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "CGAERRBAR": ("CGAERRBAR",),
    "UTMALDG": ("UTMALDG",),
    "R2P": ("R2P",),
    "MUFU.EX2": ("MUFU.EX2",),
    "BAR.SYNC": ("BAR.SYNC",),
}
# The 3-WG renderings' spill counts MEASURED on the branch's own toolchain (cutlass-dsl 4.8.0 + the CUDA 13.5 ptxas, sm_107a, 2026-10-09):
# the bias + kv_mask + padded + causal arm STL 5 / LDL 9 -- in the 56-register CORRECTION group's masked epilogue (the `_kv_empty` band
# arithmetic beside the per-row trim), not in the 128-register softmax; the streamed bias arm 0 / 0 (the two-group rendering of each arm reads
# 0 / 0).  A bound, not a literal (frost_test_utils.SPILL_TOLERANCE).  The two renderings here are the DEFAULT class's (bias + kv_mask: the
# masked resident block, the streamed ring at the chunk the 3-WG body runs); the bias-only 3-WG renderings are the option's, no default renders them.
_WG3_SPILL_PINS = {"d32_pair4_t64_bias_mask_padded_causal": (5, 9), "d32_pair8_t64_bias_streamed": (0, 0)}
_WG3_ARMS = {
    "d32_pair4_t64_bias_mask_padded_causal": {
        "__d": 32,
        "cta_mma": 1,
        "dtype_qkv": 2,
        "bias_mode": 1,
        "kv_mask_form": 1,
        "seq_kv_lens_present": True,
        "window_right": 0,
        "n_chunk": 4,
        "tile_n": 64,
    },
    "d32_pair8_t64_bias_streamed": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, "n_chunk": 8, "tile_n": 64},
}


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


@pytest.mark.parametrize("arm", list(_WG3_ARMS))
def test_the_three_group_rendering_carries_the_split_the_chain_and_the_mask_form(tmp_path, arm):
    """Each 3-WG rendering traced for sm_107a is a different kernel from its 2-WG twin, keeps a real register split (`USETMAXREG` 4: the two
    softmax INCREASE sites -- WG0's arm and the shared WG1 / WG2 arm -- + 2 DECREASE (correction and the common service-warpgroup site)), no cluster-scope drain, one more TMA issue site per
    operand (the third Q / K / V / O slab), the bit-word mask form on the masked arm, 1.5x the MUFU.EX2 sites (three softmax bodies) and the
    extra named barrier of the hand-off chain; its spills stay within the measured bound (the pins above)."""
    base = _probe(tmp_path, dict(_WG3_ARMS[arm], softmax_wgs=2), f"wg2_{arm}")
    probe = _probe(tmp_path, dict(_WG3_ARMS[arm], softmax_wgs=3), f"wg3_{arm}")
    assert (base.expect["SOFTMAX_WARPGROUPS"], probe.expect["SOFTMAX_WARPGROUPS"]) == (2, 3)
    assert (probe.expect["TOTAL_WARPS"], probe.expect["TILES_Q"], probe.expect["READ_TILE_ARRIVERS"], probe.expect["BIAS_PRELOAD_EFF"]) == (20, 3, 19, 0)
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] == 4 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert probe.stats["UTMALDG"] > base.stats["UTMALDG"], (probe.stats, base.stats)
    assert probe.stats["BAR.SYNC"] > base.stats["BAR.SYNC"], "the chain's second link (named barrier 9) adds named-barrier sites"
    assert probe.stats["MUFU.EX2"] == base.stats["MUFU.EX2"] * 3 // 2, (probe.stats["MUFU.EX2"], base.stats["MUFU.EX2"])
    if "mask" in arm:
        assert probe.stats["R2P"] > 0, "the masked arm keeps the bit-word form"
    stl, ldl = _WG3_SPILL_PINS[arm]
    assert_no_new_spills(probe.stats, {"STL": stl, "LDL": ldl}, tag=f"{arm} softmax_wgs 3: ")


# --------------------------------------------------------------------------- tier 3: GPU (Rubin) -- the pair item's degenerate matrix at three groups

# The public ``softmax_wgs=3`` OPTION on the bias-only pair item (the class the measurement rejected: two cells -- a ring wrap with a dead
# chunk tail at a chunk that is not a multiple of three, a zero-length row inside a chunk under the band + padding); TILE_N 64 -> n_kv = S_kv / 64.
_CASES = {
    "nkv4_wrap_chunk5_dead_tail_one": dict(b=5, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=5),
    "nkv8_two_wraps_chunk3": dict(b=6, h=2, h_kv=2, s_q=512, s_kv=512, n_chunk=3),
    "causal_padded_zero_row_in_chunk": dict(b=6, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=3, causal=True, kv_lens=[256, 0, 100, 256, 0, 200]),
}
# The DEFAULT class, KNOB-LESS (no seam, no knob: the pick -> the record -> the plan): bias + the bool kv_mask on the pair item (a batch-broadcast
# bias on >= 2 flattened rows), dense KV lengths, ``s_kv >= 640`` with ``b_flat <= 3072`` -- the 32-wide tile at S 1024 (above the three-slab
# resident budget: the ring) and the 64-wide tile at S 768 (no cell ran the d64 default before).  The record read off the loaded module must
# carry ``softmax_wgs`` 3 at TILE_N 64 on the pair item; the two-group twin is the seam's.
_KNOB_LESS_CASES = {
    "streamed_ring_1024_chunk6": dict(b=6, h=2, h_kv=2, s_q=1024, s_kv=1024),
    "d64_pair_both_768_dense": dict(b=6, h=2, h_kv=2, s_q=768, s_kv=768, d=64),
}


def _case_inputs(case, dtype, *, kv_mask_p=None, table=None):
    return _s13._case_inputs(case, dtype, kv_mask_p=kv_mask_p, table=_CASES if table is None else table)


def _spy_loaded_records(monkeypatch):
    """Record the ``TemplateParams`` of every kernel module the adapter loads (the record the EXECUTED plan carries; a memoised load still
    passes through the spy): the compiled record is not on the graph's public surface, so a knob-less cell reads its form here."""
    from cudnn.sdpa.fwd import api_dsl

    seen = []
    real = api_dsl._load_sm100_kernel_module

    def spy(flavor, params, *a, **kw):
        seen.append(params)
        return real(flavor, params, *a, **kw)

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", spy)
    return seen


def _run_tile_n_64(q, k, v, *, scale, bias=None, kv_mask=None, sdpa_kwargs=None, seq_len_kv=None, seq_len_q=None, sink=None, stats=True, cga=None):
    """``_s13._run`` with the KV tile width pinned to 64 through the public autotune record (``cudnn.knob_type.TILE_N``): the heuristics pick
    128 for a bias-only graph at S_kv % 128 == 0 >= 384, and the arm is served at 64 only.  Two sentinel-filled launches (the two-launch delta)."""
    import cudnn
    from types import SimpleNamespace

    g, vp, o, o_gpu, stats_gpu = _s13._build(
        q, k, v, scale=scale, bias=bias, kv_mask=kv_mask, sdpa_kwargs=sdpa_kwargs, seq_len_kv=seq_len_kv, seq_len_q=seq_len_q, sink=sink, stats=stats
    )
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _s13._frost_plans(g)
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {_s13._ROW} plan in {names}"
    engine_id, record = g.get_engine_and_knobs_at_index(idx)
    record = dict(record)
    record[cudnn.knob_type.TILE_N] = 64
    if cga is not None:
        record[cudnn.knob_type.TILE_CGA_M] = cga
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
    assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    return dict(o=o_gpu, stats=stats_gpu, knobs=SimpleNamespace(cga=cga or 1, tile_n=64), names=names)


def _run(q, k, v, kw, c, _pins, *, softmax_wgs):
    _pins(softmax_wgs=softmax_wgs, n_chunk=c.get("n_chunk"), mode=c.get("mode"))
    return _run_tile_n_64(q, k, v, **kw)


@requires_rubin
@pytest.mark.parametrize("case", ["nkv4_wrap_chunk5_dead_tail_one", "causal_padded_zero_row_in_chunk"])
def test_three_groups_match_the_fp64_oracle_and_are_bitwise_the_two_group_kernel(case, _pins):
    """The public option (``softmax_wgs=3`` on the bias-only pair item): the pair item matches the fp64 oracle and is BITWISE the two-group
    kernel on O and Stats: a row's softmax runs the same instruction stream whichever group owns it; the step's grouping of rows does not
    touch a row's arithmetic (keyless rows exactly 0 / -inf, two launches bitwise inside `_s13._run`)."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    three = _run(q, k, v, kw, c, _pins, softmax_wgs=3)
    _s13._check(three, ref)
    two = _run(q, k, v, kw, c, _pins, softmax_wgs=2)
    assert torch.equal(three["o"], two["o"]) and torch.equal(three["stats"], two["stats"]), f"{case}: softmax_wgs 3 differs from 2"


@requires_rubin
@pytest.mark.parametrize("case", ["nkv8_two_wraps_chunk3", "streamed_ring_1024_chunk6", "d64_pair_both_768_dense"])
def test_three_groups_compose_with_the_kv_mask(case, _pins, monkeypatch):
    """The bit word under three groups: the option on the ring wrap (the seam's 3 against the seam's 2), and the DEFAULT class knob-less on both
    tiles -- the plan takes the pair item at TILE_N 64 with ``softmax_wgs`` 3 by the pick alone (read off the loaded record), matches the fp64
    oracle and is bitwise the two-group twin."""
    if case in _KNOB_LESS_CASES:
        c = _KNOB_LESS_CASES[case]
        q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_KNOB_LESS_CASES)
        seen = _spy_loaded_records(monkeypatch)
        _pins()  # knob-less: every seam off
        three = _s13._run(q, k, v, **kw)
        records = {(p.softmax_wgs, p.tile_n, p.n_chunk >= 1, p.bias_mode != 0, p.kv_mask_form != 0) for p in seen}
        assert records == {(3, 64, True, True, True)}, f"{case}: the knob-less plan did not take the three-group pair body at TILE_N 64: {records}"
        assert three["knobs"].tile_n == 64 and three["knobs"].cga == 1, three["knobs"]
        _s13._check(three, ref)
        _pins(softmax_wgs=2)
        two = _s13._run(q, k, v, **kw)
    else:
        c = _CASES[case]
        q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
        three = _run(q, k, v, kw, c, _pins, softmax_wgs=3)
        _s13._check(three, ref)
        two = _run(q, k, v, kw, c, _pins, softmax_wgs=2)
    assert torch.equal(three["o"], two["o"]) and torch.equal(three["stats"], two["stats"]), f"{case}: softmax_wgs 3 differs from 2"
