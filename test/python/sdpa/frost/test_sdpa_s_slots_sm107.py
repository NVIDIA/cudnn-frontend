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
the sweep seam's value INTO the record.  Tiers: CPU contract (the TMEM map gains the slot axis at 2 and is untouched at 1, the typed raises off
the domain -- TILE_N 128, d128, a quantized dtype, 3 slots -- the pick, two records -> two modules), host rendering pins (sm_107a trace-compile:
the one-slot renderings are byte-identical to the committed md5 records, the two-slot rendering differs, keeps the register split, no cluster
drain, no new spills), GPU (Rubin): the fp64 oracle at 2 slots over the ring-wrap axis n_kv = 1 / 2 / 3 / 4 / 6 / 8 / 10 (TILE_N 64), the dead-sub-tile quiet body, the
padded tail and cga2 included, bitwise 2 == 1.
"""

import dataclasses
import re
from pathlib import Path

import pytest
import torch

from frost_test_utils import assert_no_new_spills, requires_rubin, run_sass_probe, sass_probe_source
import test_sdpa_bias_sm107 as _s13
import test_sdpa_bias_streamed_sm107 as _s15

pytestmark = [pytest.mark.L0]
_RENDERINGS = Path(__file__).resolve().parent / "renderings"


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

    assert [f.name for f in dataclasses.fields(TemplateParams)][
        -1
    ] == "s_slots" and TemplateParams().s_slots == 1, "append-only; 1 renders every record byte-identically"
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


def test_the_seam_lands_in_the_record_verbatim(_pins):
    """The pick returns the pinned value VERBATIM (2 on a TILE_N 128 shape too): a plan off the arm's domain is a TYPED factory raise, never
    a silent one-slot rendering under a two-slot label -- the trap of a seam the module cache cannot see."""
    from cudnn.sdpa.fwd import config_sm107 as c

    _pins(s_slots=2)
    assert c.sm107_s_slots_pick(32, 64) == 2 and c.sm107_s_slots_pick(32, 128) == 2 and c.sm107_s_slots_pick(64, 64) == 2
    _pins(s_slots=None)
    assert c.sm107_s_slots_pick(32, 64, has_bias=True) == 1 and c.sm107_s_slots_pick(32, 64) == 2


@pytest.mark.parametrize("d", [32, 64])
def test_the_tmem_map_gains_the_slot_axis_at_two_and_is_untouched_at_one(d):
    """S[qs][p] = (qs x S_SLOTS + p) x TILE_N with P in each slot's tail, O after the LAST slot, STATS at 544: the map the kernel renders."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    make = c.make_cfg_d32 if d == 32 else c.make_cfg_d64
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


def test_kernel_carries_the_two_slot_arm_as_one_const_expr_axis():
    """Every seam folds on ``_S2`` / a trace-time ``s_idx is None`` ternary; the one-slot spelling of the barrier indices and the TMEM bases is
    kept verbatim (the byte-identity pins below are the proof; this is the static detector)."""
    code = (Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "kernels" / "sm107" / "prefill_d128_f16.py").read_text()
    assert "_S2: bool = CFG.S_SLOTS == 2" in code
    # the factory call spans lines since the bias ring's multicast form added its release-count keyword; the slot ring is its first keyword
    assert "make_classic_bars(\n        CFG,\n        s_stages=CFG.TILES_Q * CFG.S_SLOTS," in code
    for fn in ("def _mma_tile_s2(", "def _mma_item_pair_s2(", "def _mma_bmm1_s2(", "def _mma_bmm2_s2(", "def _bmm2_ready_stage0("):
        assert fn in code, fn
    # the two slot-indexed rings are initialised over EVERY stage
    assert "for p in cutlass.range_constexpr(CFG.S_SLOTS):\n                    bars.mb_bmm1_done[qs * CFG.S_SLOTS + p].init()" in code
    assert "bars.mb_bmm2_ready[(qs * CFG.S_SLOTS + p) * CFG.N_BMM2_CHUNKS + c].init()" in code
    # the softmax bodies wait the slot's bmm1_done and arrive the slot's bmm2_ready; the cursor advances once per body
    # ... the two live bodies AND the dead-sub-tile quiet body (a dead sub-tile's MMAs are issued like a live one's)
    assert code.count("bars.mb_bmm1_done[sub_tile_id if s_idx is None else sub_tile_id * CFG.S_SLOTS + s_idx].wait(") == 3
    assert code.count("advance(PipelineState(idx=s_idx, phase=bmm1_phase), CFG.S_SLOTS)") == 3
    assert "_r = _softmax_kv_body_quiet(sub_tile_id, tmem_ptr_i32, bars, bmm1_phase, stat_empty_phase, leader_cta_id, s_idx)" in code
    # every _softmax_kv_body call passes the cursor by KEYWORD (the signature carries tail_premask before it)
    assert code.count("s_idx=s_idx,") == 5 and not re.search(r"track_true_max,\n\s+s_idx,\n", code)
    # no pre-slot literal index survives on the correction ARRIVE sites (the one-slot MMA pair body WAITS `qs * N_BMM2_CHUNKS + 0`: its two-slot
    # twin `_mma_item_pair_s2` is a separate function, so that spelling is the one-slot arm, kept verbatim)
    assert not re.search(r"mb_bmm2_ready\[qs \* CFG\.N_BMM2_CHUNKS \+ 0\]\.arrive\(", code)
    assert "mb_bmm2_ready[_bmm2_ready_stage0(qs, s_corr)].arrive(" in code
    # the MMA warp does NOT wait its own bmm2_done before a BMM1 (the in-order tensor pipe orders the two MMAs)
    s2 = code[code.index("def _mma_bmm1_s2(") : code.index("def _softmax_kv_body(")]
    assert "mb_bmm2_done" in s2 and ".mb_bmm2_done[" in s2 and not re.search(r"mb_bmm2_done\[[^\]]*\]\.wait\(", s2)


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


def _record(name):
    f = _RENDERINGS / name
    assert f.is_file(), f
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"fwd_d128_file sm_107a (\S+) rc=0 cubin_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def _skip_unless_record_dsl(dsl):
    from cudnn.frost.buffers import cutedsl_state

    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: the cubin is a function of the DSL build")


@pytest.mark.parametrize("arm", list(_s15._RESIDENT_RECORD_ARMS))
def test_the_one_slot_resident_renderings_are_byte_identical_to_the_committed_record(tmp_path, arm):
    """With the field at its default every resident rendering is byte-identical to its committed md5 record: the arm adds no instruction to
    the one-slot kernel (every seam folds; the cursors are None and never loop-carried)."""
    dsl, want = _record("md5_fwd_d128_file_bias_resident_sm_107a.txt")
    _skip_unless_record_dsl(dsl)
    probe = _probe(tmp_path, dict(_s15._RESIDENT_RECORD_ARMS[arm]), f"s1_{arm}")
    assert probe.expect["S_SLOTS"] == 1
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the record's {want[arm]} -- the one-slot rendering changed"


@pytest.mark.parametrize("arm", ["d32_dense_bf16", "d64_padded_bf16_cga2"])
def test_the_one_slot_plain_renderings_are_byte_identical_to_the_committed_record(tmp_path, arm):
    dsl, want = _record("md5_fwd_d128_file_bias_off_sm_107a.txt")
    _skip_unless_record_dsl(dsl)
    probe = _probe(tmp_path, dict(_s13._RECORD_ARMS[arm]), f"s1_{arm}")
    assert probe.expect["S_SLOTS"] == 1
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the record's {want[arm]} -- the one-slot rendering changed"


@pytest.mark.parametrize(
    "d, extra",
    [
        (32, {"bias_mode": 1, "kv_mask_form": 1, "tile_n": 64, "n_chunk": 4}),
        (32, {"bias_mode": 2, "kv_mask_form": 1, "tile_n": 64, "bias_stages": 2}),
        (32, {"tile_n": 64}),
        (64, {"bias_mode": 1, "kv_mask_form": 1, "tile_n": 64, "n_chunk": 4}),
        # the two head-side arms the GPU tier reaches only at their defaults: the pre-masked padded tail (the 128-wide default; requested at
        # TILE_N 64 here) and the dead-sub-tile quiet body (default-on where a dead sub-tile exists) -- both walk the slot cursor
        (32, {"seq_kv_lens_present": True, "padded_tail_form": 1, "tile_n": 64}),
        (32, {"seq_kv_lens_present": True, "dead_subtile_quiet": 1, "tile_n": 64}),
    ],
    ids=["d32_pair4_both_t64", "d32_one_both_t64_ring2", "d32_plain_t64", "d64_pair4_both_t64", "d32_padded_premask_t64", "d32_padded_quiet_t64"],
)
def test_the_two_slot_rendering_differs_keeps_the_split_and_spills_no_more(tmp_path, d, extra):
    """The two-slot rendering is a different cubin (the slot cursors, the widened rings, the re-ordered MMA issue), keeps the per-role
    register split (USETMAXREG > 0), no cluster-scope drain (CGAERRBAR 0), the same TMA issue-site count and the same number of MMA
    instructions (the same MMAs in a different order) and spills no more than the one-slot rendering of the same arm."""
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

# n_kv tiles at TILE_N 64 = 1, 2, 3, 4, 8 (S_kv 64 / 128 / 192 / 256 / 512) + the bring-up cell (384: 6 tiles, the cursor wraps three times per
# step), the pair item (a batch-broadcast bias, B_flat >= 2) and the one-batch-entry item (a per-batch bias), the streamed ring at TILE_N 64,
# causal + padded rows, a dead sub-tile (an odd chunk tail), the 64-wide tile, B > 1 everywhere.  Every case carries a kv_mask (0.3 False), so
# the heuristics pick TILE_N 64; the plain cells take the width through the public TILE_N knob.
_CASES = {
    "pair_both_384_bringup": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4),
    "pair_both_nkv1_64": dict(b=4, h=2, h_kv=2, s_q=64, s_kv=64, n_chunk=4),
    "pair_both_nkv2_128": dict(b=4, h=2, h_kv=2, s_q=128, s_kv=128, n_chunk=4),
    "pair_both_nkv3_192_odd_tail": dict(b=3, h=2, h_kv=2, s_q=192, s_kv=192, n_chunk=4),
    "pair_both_nkv4_256": dict(b=4, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=4),
    "pair_both_nkv8_512_causal_padded": dict(b=4, h=2, h_kv=2, s_q=512, s_kv=512, n_chunk=2, causal=True, kv_lens=[512, 0, 300, 512]),
    "one_both_per_batch_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, bias="b"),
    "one_both_ring_1024": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024, mode=2),
    "d64_pair_both_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, d=64, n_chunk=4),
    "d64_pair_both_640_causal": dict(b=4, h=2, h_kv=2, s_q=640, s_kv=640, d=64, n_chunk=2, causal=True),
}
# The plain cells also cover the dead-sub-tile QUIET body (the default where S_q % 256 is in 1..128 on the one-batch-entry item: S 384 /
# 300 / 640), which walks the SAME slot cursor as the live bodies; S 300 adds the padded tail (TILE_N 64 keeps the legacy software-max
# tail body) and the cga2 cell the non-leader CTA's dead sub-tiles + the leader-collected bmm2_ready counts.  ``knob_tile_n`` None = the
# SERVED pick (no TILE_N knob: the width rule's ``s_kv % 128`` in 1..64 lands on 64 and the slot rule on 2 -- the measured default's own
# cells, S 300 dense and with per-batch KV lengths), else the public TILE_N 64 knob.
_PLAIN_T64_CASES = {
    "plain_t64_nkv3_192": dict(b=2, h=2, h_kv=2, s_q=192, s_kv=192),
    "plain_t64_nkv8_512_causal": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512, causal=True),
    "plain_t64_384_quiet": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384),
    "plain_t64_300_quiet_padded_tail": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300),
    "plain_300_served_pick": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, knob_tile_n=None),
    "plain_300_served_pick_kv_lens": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, kv_lens=[300, 200], knob_tile_n=None),
    "plain_t64_640_quiet_causal": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, causal=True),
    "plain_t64_384_quiet_cga2": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, cga=2),
    "d64_plain_t64_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, d=64),
    "d64_plain_300_served_pick": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, d=64, knob_tile_n=None),
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


def _run(q, k, v, kw, c, _pins, *, s_slots, tile_n=None, softmax_wgs=None):
    _pins(s_slots=s_slots, mode=c.get("mode"), n_chunk=c.get("n_chunk"), softmax_wgs=softmax_wgs)
    return _run_knobbed(q, k, v, kw, tile_n=tile_n)


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES))
def test_two_slots_match_the_fp64_oracle_and_are_bitwise_one_slot(case, _pins):
    """At two slots the kernel matches the fp64 oracle (bias AND kv_mask applied) over the ring-wrap axis and is BITWISE the one-slot kernel
    on O and Stats: the same instructions in the same order, only the slot each kv tile's S / P lives in alternates.  The two-slot arm is
    knob-less (at the two cells whose knob-less pick is THREE softmax warp groups -- the bias + kv_mask pair item from S_kv 640 -- the
    two-slot record DEGRADES it to two: this call would raise typed otherwise); the one-slot arm is pinned to the same two groups so the
    bitwise comparison is like for like (the three-group one-slot body is its own module's business)."""
    c = _CASES[case]
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_CASES)
    two = _run(q, k, v, kw, c, _pins, s_slots=2)
    _s13._check(two, ref)
    one = _run(q, k, v, kw, c, _pins, s_slots=None, softmax_wgs=2)
    assert torch.equal(two["o"], one["o"]) and torch.equal(two["stats"], one["stats"]), f"{case}: S_SLOTS 2 differs from 1"


@requires_rubin
@pytest.mark.parametrize("case", list(_PLAIN_T64_CASES))
def test_two_slots_on_the_plain_tile_n_64_body(case, _pins):
    """The plain (no bias, no mask) body at TILE_N 64 -- through the public knob, or the served pick itself at S 300 (``knob_tile_n`` None:
    the width rule lands on 64 and the measured slot rule on 2 with NO pin at all): the oracle sees a zero bias (s + 0 == s exactly); the
    quiet cells run the dead sub-tile's QUIET body with two slots (the cursor threaded through it), S 300 the padded legacy tail, the
    kv_lens cell the per-batch-length arm.  The knob-less plan (the default = two slots here) is BITWISE the one-slot kernel pinned
    through the seam."""
    from cudnn.sdpa.fwd import config_sm107 as cfg107

    c = _PLAIN_T64_CASES[case]
    tile_n = c.get("knob_tile_n", 64)
    assert cfg107.sm107_s_slots_pick(c.get("d", 32), 64) == 2, "the measured default the knob-less arm below renders"
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, table=_PLAIN_T64_CASES)
    ref = _s13._ref_of(c, q, k, v, {**kw, "bias": torch.zeros_like(kw["bias"])})
    kw = {**kw, "bias": None}
    two = _run(q, k, v, kw, c, _pins, s_slots=None, tile_n=tile_n)
    _s13._check(two, ref)
    one = _run(q, k, v, kw, c, _pins, s_slots=1, tile_n=tile_n)
    assert torch.equal(two["o"], one["o"]) and torch.equal(two["stats"], one["stats"]), f"{case}: S_SLOTS 2 (the default) differs from 1"


@requires_rubin
def test_two_slots_at_tile_n_128_is_a_typed_decline(_pins):
    """The seam at 2 on a TILE_N 128 plan: the factory raises typed through the plan build (never a one-slot kernel under a two-slot label)."""
    c = dict(b=2, h=2, h_kv=2, s_q=256, s_kv=256)
    q, k, v, kw, _ref = _s13._case_inputs("plain", torch.bfloat16, table={"plain": c})
    kw = {**kw, "bias": None}
    _pins(s_slots=2)
    with pytest.raises(Exception, match="TMEM|S_SLOTS|TILE_N|s_slots"):
        _run_knobbed(q, k, v, kw, tile_n=128)
