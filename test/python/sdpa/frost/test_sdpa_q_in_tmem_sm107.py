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
is a TemplateParams FIELD (appended, default 0; ``config_sm107.sm107_q_in_tmem_pick`` returns 0 until the arm is measured, the sweep seam's
value verbatim): the template-module cache keys on the record alone.  Tiers: CPU contract (the field, the seam, the pick, the TMEM map gains
the Q slots at 1 and is untouched at 0, the typed raises off the domain -- d128, a quantized dtype, cga2, the 64-row tile, three warp groups --
two records -> two modules, the kernel's one const_expr axis), host rendering pins (sm_107a trace-compile: the SMEM-Q renderings are
byte-identical to the committed md5 records, the TMEM-Q rendering differs, carries ``TILES_Q x Q_COLS / 4`` UTCCP instructions and the same
number of MMA instructions, keeps the register split, no cluster drain, no new spills), GPU (Rubin): the fp64 oracle at ``q_in_tmem`` 1 over the
ring-wrap axis and every item form the small tiles serve, bitwise the SMEM-Q kernel.
"""

import dataclasses
import re
from pathlib import Path

import pytest
import torch

from frost_test_utils import assert_no_new_spills, requires_rubin, run_sass_probe, sass_probe_source
import test_sdpa_bias_sm107 as _s13
import test_sdpa_bias_streamed_sm107 as _s15
import test_sdpa_s_slots_sm107 as _s2

pytestmark = [pytest.mark.L0]
_RENDERINGS = Path(__file__).resolve().parent / "renderings"
_KERNEL = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "kernels" / "sm107" / "prefill_d128_f16.py"


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


def test_the_field_is_appended_the_seam_is_private_and_the_pick_is_zero_until_measured():
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
            assert c.sm107_q_in_tmem_pick(d, dtype_qkv=dt) == 0, "unmeasured: the A/B writes the rule"
    assert c.sm107_q_in_tmem_pick(32, request=1) == 1 and c.sm107_q_in_tmem_pick(32, request=0) == 0
    # the seam is NOT part of the module-cache key tuple: its value lands IN the record
    assert c.sweep_seam_state() is None


def test_the_seam_lands_in_the_record_verbatim(_pins):
    from cudnn.sdpa.fwd import config_sm107 as c

    _pins(q_in_tmem=1)
    assert (
        c.sm107_q_in_tmem_pick(32) == 1 and c.sm107_q_in_tmem_pick(64, dtype_qkv=3) == 1 and c.sm107_q_in_tmem_pick(128) == 1
    ), "verbatim -- the factory declines off the domain"
    _pins(q_in_tmem=None)
    assert c.sm107_q_in_tmem_pick(32) == 0


@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize(
    "extra",
    [dict(bias_mode=1, n_chunk=4), dict(bias_mode=1, kv_mask_form=1, n_chunk=4, tile_n=64, s_slots=2), dict(bias_mode=2, bias_stages=2), dict()],
    ids=["pair4_resident_t128", "pair4_both_t64_2slots", "one_ring2_t128", "plain_t128"],
)
def test_the_tmem_map_gains_the_q_slots_at_one_and_moves_nothing_at_zero(d, extra):
    """Q_COLS = TILE_K x BPE / 4 columns per sub-tile after the LAST O slot (320 / 336 at D 32, 384 / 416 at D 64), STATS at 544, every other slot
    where it was; the SMEM budget (the resident tiles, the ring depth, the K / V ring) does not move -- the arm costs TMEM columns only."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    make = c.make_cfg_d32 if d == 32 else c.make_cfg_d64
    if d == 64 and extra.get("bias_mode") == 2:
        pytest.skip("the d64 one-batch-entry 2-stage ring never fits at cga1 (a pre-existing budget fact)")
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


def test_the_sixty_four_row_tile_declines_the_arm_through_the_seam(monkeypatch):
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    monkeypatch.setattr(c, "_TILE_M_OVERRIDE_FOR_SWEEPS", 64)
    with pytest.raises(ValueError, match="64x128b"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=4, q_in_tmem=1))


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


def test_kernel_carries_the_arm_as_one_const_expr_axis():
    """Every seam folds on ``_Q_TS``: the UTCCP helper exists, every MMA-side ``mb_q_full`` wait is followed by the folded copy, every BMM1
    site is the ``mma_ts`` / ``mma_ss`` fold, the four step-epilogue ``mb_q_empty`` commits fold OUT under the arm (the moved site), and no
    other ``mb_q_empty`` arrive exists (the byte-identity pins below are the proof; this is the static detector)."""
    code = _KERNEL.read_text()
    assert "_Q_TS: bool = CFG.Q_IN_TMEM == 1" in code
    assert "def _q_to_tmem(tmem_q_i32, desc_q, bars, qs: cutlass.Constexpr[int], mcast_mask, anchor):" in code
    assert "_Q_UTCCP_N_CALLS = LAYOUT.Q_COLS // _Q_UTCCP_COLS_PER_CALL" in code
    helper = code[code.index("def _q_to_tmem(") : code.index("def _mma_warp_quiet(")]
    assert (
        helper.count("if nvvm.elect_sync():") == 1
        and "nvvm.tcgen05_cp(" in helper
        and "bars.mb_q_empty[qs].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)" in helper
    )
    assert "Tcgen05CpShape.SHAPE_128X128B" in helper
    # ONE loop-carried descriptor: a runtime loop (never an unrolled `base + off_i` batch, which spilled the 40-register MMA warp at D 64)
    assert "for i in cutlass.range(cutlass.Int32(0), cutlass.Int32(_Q_UTCCP_N_CALLS), 1, unroll=1):" in helper and "dq = dq + cutlass.Int64(1)" in helper
    assert "desc_opaque(desc_q, anchor)" in helper and "range_constexpr" not in helper
    # 7 MMA-side Q waits, each followed by the folded copy
    waits = re.findall(
        r"bars\.mb_q_full\[(0|1|qs)\]\.wait\(q_full_phase, spin=SPIN_RING_WAITS\)\n\s+if cutlass\.const_expr\(_Q_TS\):\n\s+_q_to_tmem\(tmem_Q_i32\[\1\]", code
    )
    assert len(waits) == 7, waits
    assert len(re.findall(r"bars\.mb_q_full\[(?:0|1|qs)\]\.wait\(q_full_phase, spin=SPIN_RING_WAITS\)", code)) == 7
    # 7 BMM1 sites, every one the fold; no bare SS BMM1 survives
    assert len(re.findall(r"if cutlass\.const_expr\(_Q_TS\):\n\s+mma_ts\(bmm1_desc, ", code)) == 7
    assert len(re.findall(r"else:\n\s+mma_ss\(bmm1_desc, ", code)) == 7 and len(re.findall(r"^\s*mma_ss\(bmm1_desc, ", code, re.M)) == 7
    # the moved site: the four epilogue commits fold out under the arm, the helper's commit is the only other arrive on the row
    assert code.count("if cutlass.const_expr(not IS_QO_ALIAS and not _Q_TS):") == 4
    assert len(re.findall(r"mb_q_empty\[qs\]\.arrive\(", code)) == 5
    assert "mma_ss, mma_ts, mma_ts_step" in code


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


@pytest.mark.parametrize("arm", list(_s15._RESIDENT_RECORD_ARMS))
def test_the_smem_q_resident_renderings_are_byte_identical_to_the_committed_record(tmp_path, arm):
    dsl, want = _s2._record("md5_fwd_d128_file_bias_resident_sm_107a.txt")
    _s2._skip_unless_record_dsl(dsl)
    probe = _probe(tmp_path, dict(_s15._RESIDENT_RECORD_ARMS[arm]), f"q0_{arm}")
    assert probe.expect["Q_IN_TMEM"] == 0 and probe.stats["UTCCP"] == 0
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the record's {want[arm]} -- the SMEM-Q rendering changed"


@pytest.mark.parametrize("arm", ["d32_dense_bf16", "d64_dense_bf16", "d64_padded_bf16_cga2"])
def test_the_smem_q_plain_renderings_are_byte_identical_to_the_committed_record(tmp_path, arm):
    dsl, want = _s2._record("md5_fwd_d128_file_bias_off_sm_107a.txt")
    _s2._skip_unless_record_dsl(dsl)
    probe = _probe(tmp_path, dict(_s13._RECORD_ARMS[arm]), f"q0_{arm}")
    assert probe.expect["Q_IN_TMEM"] == 0 and probe.stats["UTCCP"] == 0
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the record's {want[arm]} -- the SMEM-Q rendering changed"


@pytest.mark.parametrize(
    "d, extra",
    [
        (32, {"bias_mode": 1, "n_chunk": 4}),
        (32, {"bias_mode": 1, "kv_mask_form": 1, "tile_n": 64, "n_chunk": 4, "s_slots": 2}),
        (32, {"bias_mode": 2, "bias_stages": 2}),
        (32, {}),
        (32, {"tile_n": 64, "s_slots": 2}),
        (64, {"bias_mode": 1, "n_chunk": 4}),
        (64, {}),
    ],
    ids=["d32_pair4_res_t128", "d32_pair4_both_t64_s2", "d32_one_ring2_t128", "d32_plain_t128", "d32_plain_t64_s2", "d64_pair4_res_t128", "d64_plain_t128"],
)
def test_the_tmem_q_rendering_differs_carries_the_utccp_calls_keeps_the_split_and_spills_no_more(tmp_path, d, extra):
    """The TMEM-Q rendering is a different cubin, carries exactly TILES_Q UTCCP instructions (one per sub-tile: the copy is a runtime loop over one
    loop-carried descriptor, Q_COLS / 4 trips per step -- 4 at D 32, 8 at D 64; an unrolled form spilled the 40-register MMA warp at D 64), the SAME number of MMA instructions (the same MMAs, A from TMEM) and TMA issue sites, keeps the per-role
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

# The bias + kv_mask cases of the two-slot module (the pair item at TILE_N 64 / 128, the one-batch-entry item, the streamed ring, causal + padded
# rows, a dead sub-tile, d64) and the plain TILE_N 64 cases: each at q_in_tmem 1 against the fp64 oracle and BITWISE the SMEM-Q kernel.  The
# TILE_N 128 pair forms the training grid picks (the resident block at 128 wide) are added here: n_kv 1 / 2 / 3 / 4 / 6 (S 128..768).
_T128_CASES = {
    "pair_bias_t128_nkv1_128": dict(b=4, h=2, h_kv=2, s_q=128, s_kv=128, n_chunk=4),
    "pair_bias_t128_nkv2_256": dict(b=4, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=4),
    "pair_bias_t128_nkv3_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4),
    "pair_bias_t128_nkv4_512_odd_tail": dict(b=3, h=2, h_kv=2, s_q=512, s_kv=512, n_chunk=4),
    "pair_bias_t128_nkv6_768_causal_padded": dict(b=4, h=2, h_kv=2, s_q=768, s_kv=768, n_chunk=8, causal=True, kv_lens=[768, 0, 500, 768]),
    "one_bias_t128_ring2_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, mode=2),
    "d64_pair_bias_t128_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, d=64, n_chunk=4),
    "d64_pair_bias_t128_768": dict(b=4, h=2, h_kv=2, s_q=768, s_kv=768, d=64, n_chunk=8),
}


def _run(q, k, v, kw, c, _pins, *, q_in_tmem, tile_n=None, s_slots=None):
    _pins(q_in_tmem=q_in_tmem, s_slots=s_slots, mode=c.get("mode"), n_chunk=c.get("n_chunk"), softmax_wgs=2)
    return _s2._run_knobbed(q, k, v, kw, tile_n=tile_n)


@requires_rubin
@pytest.mark.parametrize("case", list(_s2._CASES))
def test_q_in_tmem_matches_the_oracle_and_is_bitwise_smem_q_on_the_both_forms(case, _pins):
    """Bias AND kv_mask (the two-slot module's cases: TILE_N 64 through the heuristics' pick) at q_in_tmem 1: the fp64 oracle, and BITWISE the
    SMEM-Q kernel on O and Stats -- the same MMAs over the same operands in the same k-step order, only the A operand's home moves."""
    c = _s2._CASES[case]
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_s2._CASES)
    on = _run(q, k, v, kw, c, _pins, q_in_tmem=1)
    _s13._check(on, ref)
    off = _run(q, k, v, kw, c, _pins, q_in_tmem=0)
    assert torch.equal(on["o"], off["o"]) and torch.equal(on["stats"], off["stats"]), f"{case}: Q_IN_TMEM 1 differs from 0"


@requires_rubin
@pytest.mark.parametrize("case", list(_T128_CASES))
def test_q_in_tmem_matches_the_oracle_and_is_bitwise_smem_q_on_the_bias_forms_at_tile_n_128(case, _pins):
    """The bias-only forms at TILE_N 128 (the training grid's resident picks) over n_kv 1..6, the odd chunk tail, causal + padded rows, the
    one-batch-entry ring and d64: the oracle at q_in_tmem 1, bitwise the SMEM-Q kernel."""
    c = _T128_CASES[case]
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, table=_T128_CASES)
    kw = {**kw, "kv_mask": None}
    on = _run(q, k, v, kw, c, _pins, q_in_tmem=1, tile_n=128)
    _s13._check(on, ref)
    off = _run(q, k, v, kw, c, _pins, q_in_tmem=0, tile_n=128)
    assert torch.equal(on["o"], off["o"]) and torch.equal(on["stats"], off["stats"]), f"{case}: Q_IN_TMEM 1 differs from 0"


@requires_rubin
@pytest.mark.parametrize(
    "case", ["plain_t64_nkv3_192", "plain_t64_nkv8_512_causal", "plain_t64_384_quiet", "plain_t64_300_quiet_padded_tail", "d64_plain_t64_384"]
)
def test_q_in_tmem_on_the_plain_tile_n_64_body(case, _pins):
    """The plain body at TILE_N 64 with two slots (the measured default there) and the dead-sub-tile quiet body / the padded tail: the oracle at
    q_in_tmem 1, bitwise the SMEM-Q kernel (both at two slots)."""
    c = _s2._PLAIN_T64_CASES[case]
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, table=_s2._PLAIN_T64_CASES)
    ref = _s13._ref_of(c, q, k, v, {**kw, "bias": torch.zeros_like(kw["bias"])})
    kw = {**kw, "bias": None}
    on = _run(q, k, v, kw, c, _pins, q_in_tmem=1, tile_n=64, s_slots=2)
    _s13._check(on, ref)
    off = _run(q, k, v, kw, c, _pins, q_in_tmem=0, tile_n=64, s_slots=2)
    assert torch.equal(on["o"], off["o"]) and torch.equal(on["stats"], off["stats"]), f"{case}: Q_IN_TMEM 1 differs from 0"


@requires_rubin
def test_q_in_tmem_at_cga2_is_a_typed_decline(_pins):
    """The seam at 1 on a cga2 plan: the factory raises typed through the plan build (never an SMEM-Q kernel under a TMEM-Q label)."""
    c = dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, d=64, cga=2)
    q, k, v, kw, _ref = _s13._case_inputs("plain", torch.bfloat16, table={"plain": c})
    kw = {**kw, "bias": None}
    _pins(q_in_tmem=1)
    with pytest.raises(Exception, match="SELF-FILL|cga1|q_in_tmem|Q_IN_TMEM"):
        _s2._run_knobbed(q, k, v, kw, tile_n=128)
