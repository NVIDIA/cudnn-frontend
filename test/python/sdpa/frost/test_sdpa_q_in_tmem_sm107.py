# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Q resident in TMEM (``TemplateParams.q_in_tmem`` / ``Cfg.Q_IN_TMEM`` = 1) on the Rubin 32- / 64-wide f16/bf16 prefill tiles, as a measured
lever.

Today every BMM1 of the d128-file body reads Q from SMEM through its tcgen05 descriptor on every kv tile and the MMA warp releases the Q slab
(``mb_q_empty``) only after the step's LAST BMM1.  With the arm the MMA warp copies each sub-tile's Q tile SMEM -> TMEM once per n-step
(``tcgen05.cp 128x128b``, ``Q_COLS / 4`` calls through the same descriptor, one elect block) right after its ``mb_q_full`` wait, commits
``mb_q_empty`` from that block (the slab is free the moment the copy lands, so the TMA-LDG warp prefetches the next step's Q / K / V during the
current kv loop) and issues every BMM1 as an A-from-TMEM ``tcgen05.mma`` (``sm107/prefill_d128_f16.py``; the ``sm100/prefill_d512_f16.py``
Q-in-TMEM precedent).  The same MMAs over the same operands in the same k-step order: O and Stats are BITWISE the SMEM-Q kernel's.  The switch
is a TemplateParams FIELD (appended, default 0; ``config_sm107.sm107_q_in_tmem_pick`` is the MEASURED rule per body class, the sweep seam's
value verbatim): the template-module cache keys on the record alone.  Q in TMEM is a DEFAULT on three bodies (the plain 128-wide body; the
bias-only 64-wide body; the bias + kv_mask 64-wide body at D 64 without per-row lengths) and NOT a public knob: off the rule it is reachable
only through the private seam, so the cells here are the three defaults' and the knob-less chain.  Tiers: CPU contract (the field, the pick
per class, the adapter handing the rule the plan's geometry, the TMEM map gains the Q slots at 1 and is untouched at 0, the typed raises off
the domain -- d128, a quantized dtype, cga2, the 64-row tile, three warp groups -- two records -> two modules; the SMEM-Q renderings' byte
identity to the committed records is pinned by the bias / streamed modules), host rendering pins (sm_107a trace-compile: the TMEM-Q
renderings of the shipped classes differ, carry ``TILES_Q x Q_COLS / 4`` UTCCP instructions and the same number of MMA instructions, keep
the register split, no cluster drain, no new spills), GPU (Rubin): the fp64 oracle at ``q_in_tmem`` 1 on the plain 128-wide body and the
d64 both-forms body, the knob-less plan's pick, each bitwise the SMEM-Q kernel.
"""

import dataclasses

import pytest
import torch

from frost_test_utils import assert_no_new_spills, requires_rubin, run_sass_probe, sass_probe_source
import test_sdpa_bias_sm107 as _s13
import test_sdpa_s_slots_sm107 as _s2

pytestmark = [pytest.mark.L0]


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


@pytest.fixture
def _pins(monkeypatch):
    """Pin the Q form, the slot count, the bias FORM, the ring depth and the chunk length a knob-less plan gets (the config module's private,
    process-local seams); ``None`` = the pick."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(q_in_tmem=None, s_slots=None, mode=None, stages=None, n_chunk=None, softmax_wgs=None):
        monkeypatch.setattr(c, "_Q_IN_TMEM_OVERRIDE_FOR_SWEEPS", q_in_tmem)
        monkeypatch.setattr(c, "_S_SLOTS_OVERRIDE_FOR_SWEEPS", s_slots)
        monkeypatch.setattr(c, "_SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS", softmax_wgs)
        monkeypatch.setattr(c, "_BIAS_MODE_OVERRIDE_FOR_SWEEPS", mode)
        monkeypatch.setattr(c, "_BIAS_STAGES_OVERRIDE_FOR_SWEEPS", stages)
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", n_chunk)

    return _set


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_the_field_is_appended_the_seam_is_private_and_the_pick_without_the_geometry_is_zero():
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    assert [f.name for f in dataclasses.fields(TemplateParams)][
        -1
    ] == "q_in_tmem" and TemplateParams().q_in_tmem == 0, "append-only; 0 renders every record byte-identically"
    assert c._Q_IN_TMEM_OVERRIDE_FOR_SWEEPS is None, "the seam is a per-test / per-sweep pin, never a module default"
    assert c.SMALL_D_Q_IN_TMEM == frozenset({0, 1}) and "SMALL_D_Q_IN_TMEM" in c.__all__ and "sm107_q_in_tmem_pick" in c.__all__
    assert c._CfgSm107.Q_IN_TMEM == 0
    for d in (32, 64):
        for dt in (2, 3):
            assert c.sm107_q_in_tmem_pick(d, dtype_qkv=dt) == 0, "a caller without the record's geometry (the sweep's bare call) gets the SMEM-Q form"
    assert c.sm107_q_in_tmem_pick(32, request=1) == 1 and c.sm107_q_in_tmem_pick(32, request=0) == 0
    # the seam is NOT part of the module-cache key tuple: its value lands IN the record
    assert c.sweep_seam_state() is None


def _rule(d, *, tile_n, bias, mask, wgs=2, cga=1, tile_m=128, lens=False, dense=True, dtype=2):
    from cudnn.sdpa.fwd import config_sm107 as c

    return c.sm107_q_in_tmem_pick(
        d,
        dtype_qkv=dtype,
        tile_n=tile_n,
        has_bias=bias,
        has_kv_mask=mask,
        softmax_wgs=wgs,
        cta_mma=cga,
        tile_m=tile_m,
        seq_kv_lens_present=lens,
        dense_unsplit=dense,
    )


def test_the_measured_rule_keys_on_the_records_geometry():
    """The A/B's verdict per cell class, as the rule (one node, 100 x 3 graph replay, both forms in one process, speed-up = t(SMEM-Q) / t(TMEM-Q) - 1):
    the plain 128-wide body WINS (14 / 14, +2.6..+6.8 %), the bias-only 64-wide body WINS (27 / 30, +1.2..+3.8 %), the bias + kv_mask 64-wide
    body at D 64 without per-row lengths WINS (4 / 5, +1.3..+7.1 %); the bias 128-wide body LOSES (-2.2..+0.5 %), the kv_mask bodies at D 32 and
    the padded kv_mask bodies at D 64 are a wash, a kv_mask alone and the plain 64-wide body are unmeasured -- all of those keep Q in SMEM; off the
    arm's domain the rule is 0 (never a typed factory decline from a knob-less plan)."""
    for d in (32, 64):
        for dt in (2, 3):
            assert _rule(d, tile_n=128, bias=False, mask=False, dtype=dt) == 1, "the plain 128-wide body"
            assert _rule(d, tile_n=64, bias=False, mask=False, dtype=dt) == 0, "the plain 64-wide body: unmeasured"
            assert _rule(d, tile_n=64, bias=True, mask=False, dtype=dt) == 1, "the bias-only 64-wide body (dense)"
            assert _rule(d, tile_n=64, bias=True, mask=False, lens=True, dtype=dt) == 1, "the bias-only 64-wide body (the per-row lengths leg)"
            assert _rule(d, tile_n=128, bias=True, mask=False, dtype=dt) == 0, "the bias 128-wide body LOSES"
            assert _rule(d, tile_n=128, bias=True, mask=False, lens=True, dtype=dt) == 0
            assert _rule(d, tile_n=64, bias=False, mask=True, dtype=dt) == 0, "a kv_mask alone: unmeasured"
            assert _rule(d, tile_n=64, bias=True, mask=True, lens=True, dtype=dt) == 0, "the padded kv_mask body: a wash"
    assert _rule(64, tile_n=64, bias=True, mask=True) == 1, "the bias + kv_mask 64-wide body at D 64 without per-row lengths"
    assert _rule(32, tile_n=64, bias=True, mask=True) == 0, "the same body at D 32: a wash"
    # off the domain: 0 (the validator would decline typed)
    assert _rule(32, tile_n=128, bias=False, mask=False, cga=2) == 0
    assert _rule(32, tile_n=128, bias=False, mask=False, tile_m=64) == 0
    assert _rule(32, tile_n=64, bias=True, mask=False, wgs=3) == 0
    assert _rule(32, tile_n=128, bias=False, mask=False, dense=False) == 0
    assert _rule(32, tile_n=128, bias=False, mask=False, dtype=0) == 0 and _rule(32, tile_n=128, bias=False, mask=False, dtype=1) == 0
    assert _rule(128, tile_n=128, bias=False, mask=False) == 0
    # the request and the seam stay verbatim above the rule
    from cudnn.sdpa.fwd import config_sm107 as c

    assert (
        c.sm107_q_in_tmem_pick(
            32, request=0, tile_n=128, has_bias=False, has_kv_mask=False, softmax_wgs=2, cta_mma=1, tile_m=128, seq_kv_lens_present=False, dense_unsplit=True
        )
        == 0
    )


@pytest.fixture
def _rubin_adapter(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a Rubin device on any host (the bias module's fixture)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


def _rule_of(p):
    """The rule re-derived from a RECORD's own fields -- what the adapter must have handed it."""
    return _rule(
        p.d_flavor,
        tile_n=p.tile_n,
        bias=p.bias_mode != 0,
        mask=p.kv_mask_form != 0,
        wgs=p.softmax_wgs,
        cga=p.cta_mma,
        lens=p.seq_kv_lens_present,
        dtype=p.dtype_qkv,
    )


# one cell per class of the rule (the D-dependent branch of the both-forms body on both sides; the d64 twins of the other classes are the same arithmetic)
@pytest.mark.parametrize(
    "d, kw, want",
    [
        (32, dict(s_q=384, s_kv=384), 1),  # the plain one-batch-entry body at TILE_N 128
        (32, dict(s_q=384, s_kv=384, bias=(1, 4, 384, 384)), 0),  # the resident pair block at TILE_N 128: loses, keeps Q in SMEM
        (32, dict(s_q=384, s_kv=384, bias=(1, 4, 384, 384), seq_kv_lens_present=True), 1),  # the per-row lengths leg: the 64-wide pair body
        (32, dict(s_q=384, s_kv=384, bias=(1, 4, 384, 384), kv_mask=(1, 1, 1, 384)), 0),  # bias + kv_mask at D 32: a wash
        (64, dict(s_q=384, s_kv=384, bias=(1, 4, 384, 384), kv_mask=(1, 1, 1, 384)), 1),  # bias + kv_mask at D 64, dense: wins
        (64, dict(s_q=640, s_kv=640, bias=(1, 4, 640, 640), kv_mask=(1, 1, 1, 640)), 0),  # the three-warp-group pick: off the domain, never a decline
    ],
    ids=["plain_d32", "bias_t128_d32", "bias_padded_d32", "both_d32", "both_d64", "both_d64_S640"],
)
def test_the_adapter_hands_the_rule_the_plans_geometry(_rubin_adapter, d, kw, want):
    """The knob-less plan's record carries the rule's pick, derived from the SAME facts the record itself names (re-derived here from the record).
    The batch is triangle attention's: B_flat = S flattened rows (the pair item needs >= 2 rows; a one-row batch lowers the one-batch-entry ring)."""
    api = _s13._api(d, b=kw["s_q"], h=4, **kw)
    api.check_support()
    p = api.template_params()
    geometry = (p.tile_n, p.bias_mode, p.kv_mask_form, p.softmax_wgs, p.cta_mma, p.seq_kv_lens_present, p.n_chunk)
    assert p.q_in_tmem == want == _rule_of(p), (p.q_in_tmem, want, geometry)


# the plain map at D 32 (one formula) and the 2-slot both-forms map at D 64 (the s_slots axis composed with the Q slots: two formulas in one record)
@pytest.mark.parametrize(
    "d, extra", [(32, dict()), (64, dict(bias_mode=1, kv_mask_form=1, n_chunk=4, tile_n=64, s_slots=2))], ids=["plain_t128-32", "pair4_both_t64_2slots-64"]
)
def test_the_tmem_map_gains_the_q_slots_at_one_and_moves_nothing_at_zero(d, extra):
    """Q_COLS = TILE_K x BPE / 4 columns per sub-tile after the LAST O slot (320 / 336 at D 32, 384 / 416 at D 64), STATS at 544, every other slot
    where it was; the SMEM budget (the resident tiles, the ring depth, the K / V ring) does not move -- the arm costs TMEM columns only."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    make = c.make_cfg_d32 if d == 32 else c.make_cfg_d64
    off, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, **extra))
    on, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, q_in_tmem=1, **extra))
    lo, ln = c.d128_family_tmem_layout(off), c.d128_family_tmem_layout(on)
    assert (off.Q_IN_TMEM, on.Q_IN_TMEM) == (0, 1)
    assert (
        (lo.S_OFF, lo.P_OFF, lo.O_OFF, lo.STATS_OFF, lo.TOTAL_COLS)
        == (ln.S_OFF, ln.P_OFF, ln.O_OFF, ln.STATS_OFF, ln.TOTAL_COLS)
        == (
            ln.S_OFF,
            ln.P_OFF,
            ln.O_OFF,
            544,
            576,
        )
    )
    assert ln.Q_COLS == lo.Q_COLS == d // 2 and ln.Q_OFF == lo.Q_OFF == (
        ln.O_OFF[-1] + d,
        ln.O_OFF[-1] + d + d // 2,
    ), "computed on every record, validated under the arm"
    assert ln.Q_OFF[-1] + ln.Q_COLS <= ln.STATS_OFF and all(q % 8 == 0 for q in ln.Q_OFF)
    assert (on.BIAS_RESIDENT_TILES, on.STAGES_BIAS, on.STAGES_KV, on.TILE_N, on.S_SLOTS) == (
        off.BIAS_RESIDENT_TILES,
        off.STAGES_BIAS,
        off.STAGES_KV,
        off.TILE_N,
        off.S_SLOTS,
    )
    # d128 untouched (and the arm does not fit its map: Q would end at 640)
    cfg128, _ = c.make_cfg_d128(TemplateParams(dtype_qkv=2, cta_mma=2))
    l = c.d128_family_tmem_layout(cfg128)
    assert (l.S_OFF, l.P_OFF, l.O_OFF, l.STATS_OFF, l.Q_OFF, l.Q_COLS) == ((0, 128), (64, 192), (256, 384), 544, (512, 576), 64)


def test_the_arm_off_its_domain_raises_typed():
    """d128 (the Q slots end at 640 > 544 and the rendering keeps its record), a quantized dtype, cga2 (the cluster self-fill UTCCP form), the
    64-row tile (the 64x128b UTCCP shape is unpinned), three softmax warp groups and a value outside {0, 1} are raises that name the failure."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    with pytest.raises(ValueError, match="32- / 64-wide tiles only"):
        c.make_cfg_d128(TemplateParams(cta_mma=1, dtype_qkv=2, q_in_tmem=1))
    with pytest.raises(ValueError, match="f16/bf16"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=0, q_in_tmem=1))
    with pytest.raises(ValueError, match="SELF-FILL"):
        c.make_cfg_d64(TemplateParams(cta_mma=2, dtype_qkv=2, q_in_tmem=1))
    with pytest.raises(ValueError, match="two-sub-tile"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, bias_mode=1, kv_mask_form=1, n_chunk=6, tile_n=64, softmax_wgs=3, q_in_tmem=1))
    with pytest.raises(ValueError, match="q_in_tmem must be 0"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, q_in_tmem=2))
    cfg, _ = c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, q_in_tmem=1))
    with pytest.raises(ValueError, match="64x128b"):
        c._validate_q_in_tmem(dataclasses.replace(cfg, TILE_M=64), "probe")
    with pytest.raises(ValueError, match="STATS"):
        c._validate_q_in_tmem(dataclasses.replace(cfg, TILE_K=128, TILE_O=128, Q_SWZ_BYTES=128, K_SWZ_BYTES=128), "probe")  # the fit predicate, explicitly


def test_two_records_are_two_modules_and_the_adapter_reads_the_pick(_pins):
    """The module cache keys on the record: a TMEM-Q record loads a SECOND module beside the SMEM-Q one in the same process (the A/B's
    precondition); the module's constants derive from the column map (never a literal)."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    base = dict(cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=4)
    for d, n_calls, per_subtile, stride in ((32, 4, 4, 512), (64, 8, 8, 1024)):
        m0 = _load_sm100_kernel_module((d, d), TemplateParams(**base), rubin=True)
        m1 = _load_sm100_kernel_module((d, d), TemplateParams(**base, q_in_tmem=1), rubin=True)
        assert m0 is not m1 and (m0._Q_TS, m1._Q_TS) == (False, True)
        assert (m1.LAYOUT.Q_COLS, m1._Q_UTCCP_N_CALLS, m1._Q_UTCCP_PER_SUBTILE, m1._Q_UTCCP_SUBTILE_DESC_STRIDE) == (d // 2, n_calls, per_subtile, stride)
        assert m1._Q_UTCCP_N_CALLS * 16 * 128 == m1.CFG.TILE_M * m1.CFG.TILE_K * m1.CFG.BPE, "the calls move exactly the Q tile"
        assert m1.LAYOUT.Q_OFF == (m1.LAYOUT.O_OFF[-1] + d, m1.LAYOUT.O_OFF[-1] + d + d // 2)
    _pins(q_in_tmem=1)
    assert c.sm107_q_in_tmem_pick(32) == 1


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("Q_IN_TMEM", mod.CFG.Q_IN_TMEM)
print("TILE_N", mod.CFG.TILE_N)
print("S_SLOTS", mod.CFG.S_SLOTS)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("EXPECT_UTCCP", mod.CFG.TILES_Q if mod._Q_TS else 0)
print("UTCCP_CALLS_PER_STEP", mod.CFG.TILES_Q * mod._Q_UTCCP_N_CALLS if mod._Q_TS else 0)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "CGAERRBAR": ("CGAERRBAR",),
    "UTMALDG": ("UTMALDG",),
    "UTCHMMA": ("UTCHMMA",),
    "UTCCP": ("UTCCP",),
    "LDTM": ("LDTM",),
}


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


# the three shipped TMEM-Q rendering classes: the plain 128-wide body at both widths and the both-forms 64-wide body (two slots)
@pytest.mark.parametrize(
    "d, extra",
    [(32, {}), (64, {}), (32, {"bias_mode": 1, "kv_mask_form": 1, "tile_n": 64, "n_chunk": 4, "s_slots": 2})],
    ids=["d32_plain_t128", "d64_plain_t128", "d32_pair4_both_t64_s2"],
)
def test_the_tmem_q_rendering_differs_carries_the_utccp_calls_keeps_the_split_and_spills_no_more(tmp_path, d, extra):
    """The TMEM-Q rendering (the shipped classes) is a different cubin, carries exactly TILES_Q UTCCP instructions (one per sub-tile: the copy
    is a runtime loop over one loop-carried descriptor, Q_COLS / 4 trips per step -- 4 at D 32, 8 at D 64; an unrolled form spilled the
    40-register MMA warp at D 64), the SAME number of MMA instructions (the same MMAs, A from TMEM) and TMA issue sites, keeps the per-role
    register split (USETMAXREG > 0), no cluster-scope drain (CGAERRBAR 0) and spills no more than the SMEM-Q rendering of the same arm."""
    base = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, **extra}, f"q0_{d}_{'_'.join(sorted(extra))}")
    probe = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, "q_in_tmem": 1, **extra}, f"q1_{d}_{'_'.join(sorted(extra))}")
    assert base.expect["Q_IN_TMEM"] == 0 and probe.expect["Q_IN_TMEM"] == 1
    assert probe.cubin_md5 != base.cubin_md5
    assert base.stats["UTCCP"] == 0 and probe.stats["UTCCP"] == probe.expect["EXPECT_UTCCP"] == 2, (probe.stats, probe.expect)
    assert probe.expect["UTCCP_CALLS_PER_STEP"] == 2 * (d // 2) // 4, probe.expect
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert probe.stats["UTMALDG"] == base.stats["UTMALDG"], (probe.stats, base.stats)
    assert probe.stats["UTCHMMA"] > 0 and probe.stats["UTCHMMA"] == base.stats["UTCHMMA"], (probe.stats, base.stats)
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} Q_IN_TMEM 1 {sorted(extra)} vs 0: ")


# --------------------------------------------------------------------------- tier 3: GPU (Rubin) -- every item form, the ring-wrap axis, bitwise

# The bias + kv_mask 64-wide body at D 64 (the rule's third default class: TILE_N 64 through the heuristics' pick, the pair item at its
# resident budget edge and on the ring under the band), each at q_in_tmem 1 against the fp64 oracle and BITWISE the SMEM-Q kernel.  (The D 32
# both-forms body is a wash and the bias 128-wide body LOSES under the arm -- rule 0, no plan path -- so neither has a cell.)
_BOTH_FORMS_CASES = {
    "d64_pair_both_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, d=64, n_chunk=4),
    "d64_pair_both_640_causal": dict(b=4, h=2, h_kv=2, s_q=640, s_kv=640, d=64, n_chunk=2, causal=True),
}
# the bias-only pair block at TILE_N 128: the rule returns 0 there (the knob-less cell below reads it off the record)
_T128_CASES = {"pair_bias_t128_nkv3_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4)}


def _run(q, k, v, kw, c, _pins, *, q_in_tmem, tile_n=None):
    _pins(q_in_tmem=q_in_tmem, mode=c.get("mode"), n_chunk=c.get("n_chunk"), softmax_wgs=2)
    return _s2._run_knobbed(q, k, v, kw, tile_n=tile_n)


@requires_rubin
@pytest.mark.parametrize("case", list(_BOTH_FORMS_CASES))
def test_q_in_tmem_matches_the_oracle_and_is_bitwise_smem_q_on_the_both_forms(case, _pins):
    """Bias AND kv_mask at D 64 (TILE_N 64 through the heuristics' pick) at q_in_tmem 1: the fp64 oracle, and BITWISE the SMEM-Q kernel on O and
    Stats -- the same MMAs over the same operands in the same k-step order, only the A operand's home moves."""
    c = _BOTH_FORMS_CASES[case]
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_BOTH_FORMS_CASES)
    on = _run(q, k, v, kw, c, _pins, q_in_tmem=1)
    _s13._check(on, ref)
    off = _run(q, k, v, kw, c, _pins, q_in_tmem=0)
    assert torch.equal(on["o"], off["o"]) and torch.equal(on["stats"], off["stats"]), f"{case}: Q_IN_TMEM 1 differs from 0"


# The plain body at TILE_N 128 -- the rule's first class (the one-batch-entry item): dense, causal, per-batch lengths with a zero-length row, d64.
_PLAIN_T128_CASES = {
    "plain_t128_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384),
    "plain_t128_512_causal": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512, causal=True),
    "plain_t128_384_padded": dict(b=3, h=2, h_kv=2, s_q=384, s_kv=384, kv_lens=[384, 0, 200]),
    "d64_plain_t128_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, d=64),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_PLAIN_T128_CASES))
def test_q_in_tmem_on_the_plain_tile_n_128_body(case, _pins):
    """The plain one-batch-entry body at TILE_N 128 (the rule's pick) incl. the causal and per-batch-length arms: the oracle at q_in_tmem 1,
    bitwise the SMEM-Q kernel."""
    c = _PLAIN_T128_CASES[case]
    q, k, v, kw, _ = _s13._case_inputs(case, torch.bfloat16, table=_PLAIN_T128_CASES)
    ref = _s13._ref_of(c, q, k, v, {**kw, "bias": torch.zeros_like(kw["bias"])})
    kw = {**kw, "bias": None}
    on = _run(q, k, v, kw, c, _pins, q_in_tmem=1, tile_n=128)
    _s13._check(on, ref)
    off = _run(q, k, v, kw, c, _pins, q_in_tmem=0, tile_n=128)
    assert torch.equal(on["o"], off["o"]) and torch.equal(on["stats"], off["stats"]), f"{case}: Q_IN_TMEM 1 differs from 0"


# The knob-less padded bias cell: per-batch lengths with a zero-length row, the chunk length from the pick (no request), the 64-wide pair body.
_PICK_CASES = {"pair_bias_padded_t64_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, kv_lens=[384, 0, 200, 384])}


@requires_rubin
@pytest.mark.parametrize(
    "case, kw_extra, picked",
    [
        ("plain_t128_384", dict(bias=None), 1),
        ("pair_bias_t128_nkv3_384", dict(kv_mask=None), 0),
        ("pair_bias_padded_t64_384", dict(kv_mask=None), 1),
    ],
    ids=["plain_picks_tmem_q", "bias_t128_keeps_smem_q", "bias_padded_t64_picks_tmem_q"],
)
def test_the_knob_less_plan_takes_the_rules_pick_and_is_bitwise_its_smem_q_twin(case, kw_extra, picked, _pins):
    """The knob-less plan (no seam, the heuristics' picks everywhere) renders the form the rule names and is bitwise the explicitly pinned SMEM-Q
    form on O and Stats; the record it was loaded with names the pick."""
    table = {**_PLAIN_T128_CASES, **_T128_CASES, **_PICK_CASES}
    c = dict(table[case])
    tile_n = 64 if picked and c.get("kv_lens") is not None else 128  # the per-row-lengths leg runs the 64-wide pair body (the width rule)
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, table=table)
    if kw_extra.get("bias", 1) is None:
        ref = _s13._ref_of(c, q, k, v, {**kw, "bias": torch.zeros_like(kw["bias"])})
    kw = {**kw, **kw_extra}
    knob_less = _run(q, k, v, kw, c, _pins, q_in_tmem=None, tile_n=tile_n)
    _s13._check(knob_less, ref)
    pinned = _run(q, k, v, kw, c, _pins, q_in_tmem=0, tile_n=tile_n)
    assert torch.equal(knob_less["o"], pinned["o"]) and torch.equal(knob_less["stats"], pinned["stats"]), f"{case}: the knob-less plan differs from SMEM-Q"
    _pins()  # the seams back to the picks: the pinned twin above left the Q form at 0, and the standalone adapter below must read the RULE
    d = c.get("d", 32)
    api = _s13._api(
        d,
        b=c["b"],
        h=c["h"],
        s_q=c["s_q"],
        s_kv=c["s_kv"],
        bias=None if kw.get("bias") is None else (1, c["h"], c["s_q"], c["s_kv"]),
        seq_kv_lens_present=c.get("kv_lens") is not None,
        n_chunk=c.get("n_chunk"),
    )
    api.check_support()
    p = api.template_params()
    assert p.q_in_tmem == picked == _rule_of(p) and p.tile_n == tile_n, (p.q_in_tmem, picked, p.tile_n, p.bias_mode, p.kv_mask_form, p.softmax_wgs, p.cta_mma)
