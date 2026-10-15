# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The double-buffered S / P TMEM slot (``TemplateParams.s_slots`` / ``Cfg.S_SLOTS`` = 2) of the Rubin 32- / 64-wide f16/bf16 prefill tiles at
TILE_N 64, as a measured lever.

With two S / P slots per softmax sub-tile the MMA warp issues BMM1(k + 1) of both sub-tiles into the free slot AHEAD of BMM2(k), so the tensor
core runs it while the softmax warps are still on S(k) (``sm107/prefill_d128_f16.py`` at ``d_flavor`` 32 / 64; the ``sm100/decode_d128_f16.py``
two-slot precedent).  Every role walks a depth-2 ``PipelineState`` cursor over the slots, advanced once per non-empty kv tile; the same MMAs and
softmax steps run in the same order, so O and Stats are BITWISE the one-slot kernel's -- a pure overlap lever.  The switch is a TemplateParams
FIELD (appended, default 1; the knob-less pick ``config_sm107.sm107_s_slots_pick`` is the MEASURED rule: 2 on the plain body at TILE_N 64, 1 on
every other form): the template-module cache keys on the record alone, so a module-level seam read at module-exec time would hand a
second plan with the same record the FIRST module (an in-process A/B compares the base with itself); ``config_sm107.sm107_s_slots_pick`` puts
the sweep seam's value INTO the record.  Two slots is the DEFAULT on the PLAIN TILE_N 64 body only (the served pick at ``s_kv % 128`` in 1..64
and the public TILE_N 64 knob); on a bias form it is the arm the measurement rejected (one slot), reachable only through the seam, so the
cells here are the plain body's.  Tiers: CPU contract (the TMEM map gains the slot axis at 2 and is untouched at 1, the typed raises off the
domain -- TILE_N 128, d128, a quantized dtype, 3 slots -- the pick, two records -> two modules; the one-slot renderings' byte identity to the
committed records is pinned by the bias / streamed modules), host rendering pins (sm_107a trace-compile: the two-slot default-class renderings
differ, keep the register split, no cluster drain, no new spills), GPU (Rubin): the fp64 oracle at 2 slots on the served pick's forms (the
window / sink / bottom-right / f16 / per-batch-length arms, d64), the ring wrap (n_kv 8), the quiet body at cga2 -- every one bitwise 2 == 1.
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
    """Pin the slot count, the bias FORM, the bias ring depth and the chunk length a knob-less plan gets (the config module's private,
    process-local seams); ``None`` = the pick."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(s_slots=None, mode=None, stages=None, n_chunk=None, softmax_wgs=None):
        monkeypatch.setattr(c, "_S_SLOTS_OVERRIDE_FOR_SWEEPS", s_slots)
        monkeypatch.setattr(c, "_SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS", softmax_wgs)
        monkeypatch.setattr(c, "_BIAS_MODE_OVERRIDE_FOR_SWEEPS", mode)
        monkeypatch.setattr(c, "_BIAS_STAGES_OVERRIDE_FOR_SWEEPS", stages)
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", n_chunk)

    return _set


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_the_field_is_appended_the_seam_is_private_and_the_record_default_is_one():
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    assert [f.name for f in dataclasses.fields(TemplateParams)][-2:] == [
        "s_slots",
        "q_in_tmem",
    ] and TemplateParams().s_slots == 1, "append-only; 1 renders every record byte-identically"
    assert c._S_SLOTS_OVERRIDE_FOR_SWEEPS is None, "the seam is a per-test / per-sweep pin, never a module default"
    assert c.SMALL_D_S_SLOTS == frozenset({1, 2}) and "SMALL_D_S_SLOTS" in c.__all__ and "sm107_s_slots_pick" in c.__all__
    assert c._CfgSm107.S_SLOTS == 1
    for d in (32, 64):
        for tile_n in (64, 128):
            assert c.sm107_s_slots_pick(d, tile_n, has_bias=True, has_kv_mask=True) == 1
    assert c.sm107_s_slots_pick(32, 64, request=2) == 2 and c.sm107_s_slots_pick(32, 64, request=1) == 1


def test_the_measured_default_is_two_slots_on_the_plain_tile_n_64_body_only():
    """The knob-less pick is the measured rule (``sm107_s_slots_pick``'s docstring carries the cells): two slots on the plain body (no bias,
    no kv_mask) at TILE_N 64 of the f16/bf16 32- / 64-wide tiles -- the served pick at ``s_kv % 128`` in 1..64 (S 300: d32 +4.33 %, d64
    +6.63 %) and the public TILE_N 64 knob (+8.4..+11.9 %) -- and one slot on every form that lost or was not measured: a bias + kv_mask
    (-0.69..-5.93 %), a bias alone (mixed), a kv_mask alone (unmeasured), TILE_N 128 (no TMEM fit), the quantized dtypes and d128 (not
    wired).  The adapter threads the graph facts in (``_s_slots``); a caller without them gets the plain rule."""
    from cudnn.sdpa.fwd import config_sm107 as c

    for d in (32, 64):
        assert c.sm107_s_slots_pick(d, 64) == 2, d
        assert c.sm107_s_slots_pick(d, 64, has_bias=False, has_kv_mask=False) == 2, d
        assert c.sm107_s_slots_pick(d, 128) == 1, "TILE_N 128: 2 x 2 x 128 + O leaves no room for STATS"
        assert c.sm107_s_slots_pick(d, 64, has_bias=True) == 1, "bias alone: mixed sign, one slot"
        assert c.sm107_s_slots_pick(d, 64, has_kv_mask=True) == 1, "kv_mask alone: unmeasured, one slot"
        assert c.sm107_s_slots_pick(d, 64, has_bias=True, has_kv_mask=True) == 1, "bias + kv_mask: a loss on every cell"
        assert c.sm107_s_slots_pick(d, 64, dtype_qkv=0) == 1 and c.sm107_s_slots_pick(d, 64, dtype_qkv=1) == 1, "fp8: not wired"
    assert c.sm107_s_slots_pick(128, 64) == 1, "d128: not wired"
    assert c.sm107_s_slots_pick(32, 64, request=1, has_bias=False) == 1, "a request passes verbatim"
    # the served cell's width rule and the slot rule compose: S 300 plain -> TILE_N 64 -> 2 slots; S 384 plain -> 128 -> 1
    assert c.sm107_tile_n_pick(32, 300, has_bias=False, has_kv_mask=False) == 64 and c.sm107_tile_n_pick(32, 384, has_bias=False, has_kv_mask=False) == 128


def test_the_tmem_map_gains_the_slot_axis_at_two_and_is_untouched_at_one():
    """S[qs][p] = (qs x S_SLOTS + p) x TILE_N with P in each slot's tail, O after the LAST slot, STATS at 544: the map the kernel renders (one
    formula; D 32 here, the d64 map is the same arithmetic with a wider O slot)."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    d = 32
    make = c.make_cfg_d32
    one, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, tile_n=64, bias_mode=1, kv_mask_form=1, n_chunk=4))
    two, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, tile_n=64, bias_mode=1, kv_mask_form=1, n_chunk=4, s_slots=2))
    l1, l2 = c.d128_family_tmem_layout(one), c.d128_family_tmem_layout(two)
    assert (one.S_SLOTS, l1.S_SLOTS, l1.S_SLOT_STRIDE) == (1, 1, 0)
    assert (l1.S_OFF, l1.P_OFF, l1.O_OFF[0], l1.STATS_OFF) == ((0, 64), (32, 96), 128, 544), "the one-slot map is the map the kernel always had"
    assert (two.S_SLOTS, l2.S_SLOTS, l2.S_SLOT_STRIDE) == (2, 2, 64)
    assert (l2.S_OFF, l2.P_OFF, l2.O_OFF, l2.STATS_OFF, l2.TOTAL_COLS) == ((0, 128), (32, 160), (256, 256 + d), 544, 576)
    assert l2.O_OFF[-1] + d <= l2.STATS_OFF
    # the slot count moves nothing in SMEM: the same operands, the same resident block
    assert (two.STAGES_KV, two.BIAS_RESIDENT_TILES, two.TILE_N, two.N_BMM2_CHUNKS) == (one.STAGES_KV, one.BIAS_RESIDENT_TILES, 64, 1)
    # d128 untouched
    cfg128, _ = c.make_cfg_d128(TemplateParams(dtype_qkv=2, cta_mma=2))
    l = c.d128_family_tmem_layout(cfg128)
    assert (l.S_OFF, l.P_OFF, l.O_OFF, l.STATS_OFF, l.S_SLOTS) == ((0, 128), (64, 192), (256, 384), 544, 1)


def test_a_two_slot_record_degrades_the_knobless_group_count_to_two():
    """The measured three-group default of the bias + kv_mask pair form (S_kv >= 640) yields to a two-slot record: two slots are served with two
    groups only, and a default must never produce an invalid record.  A REQUEST of three groups passes verbatim (the factory declines it typed)."""
    from cudnn.sdpa.fwd import config_sm107 as c

    three = dict(d_flavor=32, s_kv=1024, tile_n=64, pair=True, has_bias=True, has_kv_mask=True, b_flat=4)
    assert c.sm107_softmax_wgs_pick(None, **three) == 3 and c.sm107_softmax_wgs_pick(None, **three, s_slots=1) == 3
    assert c.sm107_softmax_wgs_pick(None, **three, s_slots=2) == 2
    assert c.sm107_softmax_wgs_pick(3, **three, s_slots=2) == 3, "a request passes verbatim; the factory's typed decline is the gate"
    assert c.sm107_softmax_wgs_pick(None, **{**three, "s_kv": 384}, s_slots=2) == 2


def test_two_slots_off_the_domain_raise_typed():
    """TILE_N 128 (the map ends past STATS), 3 slots, the d128 rendering, a quantized dtype and the third softmax warp group are raises that name the failure."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    with pytest.raises(ValueError, match="TMEM"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, s_slots=2))  # TILE_N 128: 2 x 2 x 128 + O = 576 > 544
    with pytest.raises(ValueError, match="TILE_N 64 form"):
        cfg, _ = c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, tile_n=64, s_slots=2))
        c._validate_tile_n(dataclasses.replace(cfg, TILE_N=128, N_BMM2_CHUNKS=2, TILE_O=32), "probe")  # the explicit predicate, past the fit one
    with pytest.raises(ValueError, match="validated at S_SLOTS"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, tile_n=64, s_slots=3))
    with pytest.raises(ValueError, match="32- / 64-wide tiles only"):
        c.make_cfg_d128(TemplateParams(cta_mma=1, dtype_qkv=2, s_slots=2))
    with pytest.raises(ValueError):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=0, tile_n=64, s_slots=2))
    with pytest.raises(
        ValueError, match="two softmax warp groups only"
    ):  # the 3-WG pair arm keeps one slot: the two-slot issue order hand-unrolls two sub-tiles
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=32, bias_mode=1, kv_mask_form=1, n_chunk=4, tile_n=64, softmax_wgs=3, s_slots=2))


def test_two_records_are_two_modules_and_the_adapter_reads_the_pick(_pins):
    """The module cache keys on the record: a two-slot record loads a SECOND module beside the one-slot one in the same process (the A/B's
    precondition), and the adapter's record field follows the pick."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    base = dict(cta_mma=1, dtype_qkv=2, tile_n=64, bias_mode=1, kv_mask_form=1, n_chunk=4)
    m1 = _load_sm100_kernel_module((32, 32), TemplateParams(**base), rubin=True)
    m2 = _load_sm100_kernel_module((32, 32), TemplateParams(**base, s_slots=2), rubin=True)
    assert m1 is not m2 and (m1.CFG.S_SLOTS, m2.CFG.S_SLOTS, m1._S2, m2._S2) == (1, 2, False, True)
    assert (m2.LAYOUT.S0_OFF, m2.LAYOUT.S1_OFF, m2.LAYOUT.P0_OFF, m2.LAYOUT.P1_OFF, m2.LAYOUT.O0_OFF, m2.LAYOUT.S_SLOT_STRIDE) == (0, 128, 32, 160, 256, 64)
    assert (m1.LAYOUT.S0_OFF, m1.LAYOUT.S1_OFF, m1.LAYOUT.P0_OFF, m1.LAYOUT.P1_OFF, m1.LAYOUT.O0_OFF, m1.LAYOUT.S_SLOT_STRIDE) == (0, 64, 32, 96, 128, 0)
    _pins(s_slots=2)
    assert c.sm107_s_slots_pick(32, 64) == 2


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("S_SLOTS", mod.CFG.S_SLOTS)
print("TILE_N", mod.CFG.TILE_N)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("O0_OFF", mod.LAYOUT.O0_OFF)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "CGAERRBAR": ("CGAERRBAR",),
    "UTMALDG": ("UTMALDG",),
    "UTCHMMA": ("UTCHMMA",),
    "LDTM": ("LDTM",),
}


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


@pytest.mark.parametrize(
    "d, extra",
    [
        (32, {"tile_n": 64}),
        # the two head-side arms the GPU tier reaches only at their defaults: the pre-masked padded tail (the 128-wide default; requested at
        # TILE_N 64 here) and the dead-sub-tile quiet body (default-on where a dead sub-tile exists) -- both walk the slot cursor
        (32, {"seq_kv_lens_present": True, "padded_tail_form": 1, "tile_n": 64}),
        (32, {"seq_kv_lens_present": True, "dead_subtile_quiet": 1, "tile_n": 64}),
    ],
    ids=["d32_plain_t64", "d32_padded_premask_t64", "d32_padded_quiet_t64"],
)
def test_the_two_slot_rendering_differs_keeps_the_split_and_spills_no_more(tmp_path, d, extra):
    """The two-slot DEFAULT-class renderings (the plain 64-wide body: dense, the pre-masked padded tail, the quiet body) are a different cubin
    (the slot cursors, the widened rings, the re-ordered MMA issue), keep the per-role register split (USETMAXREG > 0), no cluster-scope
    drain (CGAERRBAR 0), the same TMA issue-site count and the same number of MMA instructions (the same MMAs in a different order) and
    spill no more than the one-slot rendering of the same arm."""
    base = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, **extra}, f"s1_{d}_{'_'.join(sorted(extra))}")
    probe = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, "s_slots": 2, **extra}, f"s2_{d}_{'_'.join(sorted(extra))}")
    assert base.expect["S_SLOTS"] == 1 and probe.expect["S_SLOTS"] == 2 and probe.expect["TILE_N"] == 64
    assert probe.expect["O0_OFF"] == 256 and base.expect["O0_OFF"] == 128
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert probe.stats["UTMALDG"] == base.stats["UTMALDG"], (probe.stats, base.stats)
    assert probe.stats["UTCHMMA"] > 0 and probe.stats["UTCHMMA"] == base.stats["UTCHMMA"], (probe.stats, base.stats)
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} S_SLOTS 2 {sorted(extra)} vs 1: ")


# --------------------------------------------------------------------------- tier 3: GPU (Rubin) -- the ring-wrap axis at two slots

# The plain body at TILE_N 64 -- the two-slot DEFAULT.  ``knob_tile_n`` None = the SERVED pick (no TILE_N knob: the width rule's ``s_kv % 128``
# in 1..64 lands on 64 and the slot rule on 2 -- the measured default's own cells, S 300 dense and with per-batch KV lengths, where the
# quiet body (S_q % 256 in 1..128) and the padded legacy tail run too), else the public TILE_N 64 knob: the ring wrap (n_kv 8 under the
# band) and the quiet body on the 2-CTA tile (the non-leader CTA's dead sub-tiles + the leader-collected bmm2_ready counts: its own
# rendering).
_PLAIN_T64_CASES = {
    "plain_t64_nkv8_512_causal": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512, causal=True),
    "plain_300_served_pick": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, knob_tile_n=None),
    "plain_300_served_pick_kv_lens": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, kv_lens=[300, 200], knob_tile_n=None),
    "plain_t64_384_quiet_cga2": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, cga=2),
    "d64_plain_300_served_pick": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, d=64, knob_tile_n=None),
    # The measured default reaches EVERY plain TILE_N 64 form the small-tile leg admits (only THD / paged / split-KV / PackGQA / score
    # side outputs walk past it), so the const_expr arms a dense pass proves nothing about get their own two-slot cell, each bitwise the
    # one-slot kernel: a causal left window (the item's trimmed kv range starts past tile 0 for the second 256-row item), a sink (the
    # sink fold in the two-slot epilogue), the bottom-right diagonal at S_q < S_kv (the diagonal offset), and f16 (the pick's dtype gate
    # is every f16/bf16 member, not bf16 alone) -- all through the served pick at S 300 (no TILE_N knob).
    "plain_300_served_pick_causal_window_64": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, causal=True, window_left=64, knob_tile_n=None),
    "plain_300_served_pick_sink": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, sink=True, knob_tile_n=None),
    "plain_300_served_pick_bottom_right_sq256": dict(b=2, h=2, h_kv=2, s_q=256, s_kv=300, bottom_right=True, knob_tile_n=None),
    "plain_300_served_pick_f16": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, dtype="f16", knob_tile_n=None),
}


def _run_knobbed(q, k, v, kw, *, tile_n=None):
    """``test_sdpa_bias_sm107._run`` with the public TILE_N knob beside the cga one (the sweep driver's plan dance)."""
    import cudnn

    g, vp, o, o_gpu, stats_gpu = _s13._build(
        q, k, v, **{key: kw[key] for key in ("scale", "bias", "kv_mask", "sdpa_kwargs", "seq_len_kv", "seq_len_q", "sink")}
    )
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _s13._frost_plans(g)
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {_s13._ROW} plan in {names}"
    want_cga = kw.get("cga") is not None and g.plans[idx].knobs.cga != kw["cga"]
    want_tn = tile_n is not None and g.plans[idx].knobs.tile_n != tile_n
    if want_cga or want_tn:
        engine_id, record = g.get_engine_and_knobs_at_index(idx)
        record = dict(record)
        if want_cga:
            record[cudnn.knob_type.TILE_CGA_M] = kw["cga"]
        if want_tn:
            record[cudnn.knob_type.TILE_N] = tile_n
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        o_gpu.fill_(float("nan"))
        stats_gpu.fill_(float("nan"))
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return o_gpu.clone(), stats_gpu.clone()

    o1, st1 = launch()
    o2, st2 = launch()
    assert torch.equal(o1, o2) and torch.equal(st1, st2), "two-launch delta on O / Stats"
    return dict(o=o_gpu, stats=stats_gpu, names=names)


def _run(q, k, v, kw, c, _pins, *, s_slots, tile_n=None):
    _pins(s_slots=s_slots, mode=c.get("mode"), n_chunk=c.get("n_chunk"))
    return _run_knobbed(q, k, v, kw, tile_n=tile_n)


@requires_rubin
@pytest.mark.parametrize("case", list(_PLAIN_T64_CASES))
def test_two_slots_on_the_plain_tile_n_64_body(case, _pins):
    """The plain (no bias, no mask) body at TILE_N 64 -- through the public knob, or the served pick itself at S 300 (``knob_tile_n`` None:
    the width rule lands on 64 and the measured slot rule on 2 with NO pin at all): the oracle sees a zero bias (s + 0 == s exactly); the
    quiet cells run the dead sub-tile's QUIET body with two slots (the cursor threaded through it), S 300 the padded legacy tail, the
    kv_lens cell the per-batch-length arm, the window / sink / bottom-right / f16 cells the mask, sink and dtype arms the default also
    reaches.  The knob-less plan (the default = two slots here) is BITWISE the one-slot kernel pinned through the seam."""
    from cudnn.sdpa.fwd import config_sm107 as cfg107

    c = _PLAIN_T64_CASES[case]
    tile_n = c.get("knob_tile_n", 64)
    dtype = torch.float16 if c.get("dtype") == "f16" else torch.bfloat16
    assert (
        cfg107.sm107_s_slots_pick(c.get("d", 32), 64, dtype_qkv=3 if dtype is torch.float16 else 2) == 2
    ), "the measured default the knob-less arm below renders"
    q, k, v, kw, ref = _s13._case_inputs(case, dtype, table=_PLAIN_T64_CASES)
    ref = _s13._ref_of(c, q, k, v, {**kw, "bias": torch.zeros_like(kw["bias"])})
    kw = {**kw, "bias": None}
    two = _run(q, k, v, kw, c, _pins, s_slots=None, tile_n=tile_n)
    _s13._check(two, ref)
    one = _run(q, k, v, kw, c, _pins, s_slots=1, tile_n=tile_n)
    assert torch.equal(two["o"], one["o"]) and torch.equal(two["stats"], one["stats"]), f"{case}: S_SLOTS 2 (the default) differs from 1"
