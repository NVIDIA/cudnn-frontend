# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Q tile height (``Cfg.TILE_M``) of the Rubin 32- / 64-wide f16/bf16 prefill tiles as a measured lever: 64 rows beside the 128 every
rendering has had.

A 64-row tile halves the Q / O slabs and the bias block (the pair item's resident budget 768 -> 1792 kv columns at D 32) and doubles the
work items of a q extent -- the parallelism lever at small S -- while BMM1 / BMM2 run ``m_dim = 64``.  The hardware fact the body follows:
an M = 64 ``tcgen05.mma`` accumulator lands in TMEM lanes 0-15 of each 32-lane sub-partition, lanes 16-31 are never written and every
slot keeps its column count.  The body renders the height as a LANE MAPPING of the softmax / correction groups (``TILE_M64_LANE_FORM`` in
``sm107/prefill_d128_f16.py``): form 0 (built) keeps one lane per row and leaves lanes 16-31 of every warp DEAD -- they compute on residue,
fire every barrier arrive (the init counts are lanes of a role, so the barrier table is unchanged) and store nothing; trivially right,
rejected for production (a MUFU.EX2 holds the pipe whatever the active mask), kept as the bring-up reference and the A/B lower bound.
Form 1 (the quad fragment: ``16x256b`` loads / ``16x128b`` stores, 4 threads per row) is the next session's; the module raises on it.

The height is pinned through the private, process-local seam ``config_sm107._TILE_M_OVERRIDE_FOR_SWEEPS`` (64 / 128 = ``SMALL_D_TILE_MS``;
the A/B's arm, never a plan path; the derived height stays 128), read by the ONE helper the factory, the bias budgets and the kernel's
derivations share -- and folded into the template-module cache key and the compiled-plan digest (``config_sm107.sweep_seam_state``), so
two renderings that differ only by a seam never share a module in one process.  Tiers: CPU contract (the seam moves the height, the slabs,
the block rows and the budgets together, raises outside the set, declines cga2 / THD / split / PackGQA / gate at 64, the d128 and quantized
factories ignore it, the TMEM stats width is a parameter with the record default, the seam separates module-cache entries), host rendering
pins (sm_107a trace-compile: every TILE_M 128 rendering of the two committed md5 records stays BYTE-IDENTICAL; the 64-row renderings differ,
keep the register split, no cluster drain, the same TMA issue-site count, no new spills), GPU (Rubin): the fp64 oracle at TILE_M 64 over the
ring-wrap axis n_kv = 1 / 2 / 3 / 4 / 8 with B > 1, a masked arm, the pair and the one-batch-entry items, the 64-wide tile; form 0's per-row
arithmetic is the 128-row tile's, so O and Stats are BITWISE the TILE_M 128 kernel's.
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
    """Pin the Q tile height, the K / V ring depth, the bias FORM, the bias ring depth and the chunk length a knob-less plan gets (the config
    module's private, process-local seams); ``None`` = the pick."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(tile_m=None, stages_kv=None, mode=None, stages=None, n_chunk=None):
        monkeypatch.setattr(c, "_TILE_M_OVERRIDE_FOR_SWEEPS", tile_m)
        monkeypatch.setattr(c, "_STAGES_KV_OVERRIDE_FOR_SWEEPS", stages_kv)
        monkeypatch.setattr(c, "_BIAS_MODE_OVERRIDE_FOR_SWEEPS", mode)
        monkeypatch.setattr(c, "_BIAS_STAGES_OVERRIDE_FOR_SWEEPS", stages)
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", n_chunk)

    return _set


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_the_seam_is_private_and_the_derived_height_is_128():
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c._TILE_M_OVERRIDE_FOR_SWEEPS is None, "the seam is a per-test / per-sweep pin, never a module default"
    assert c.SMALL_D_TILE_MS == frozenset({64, 128})
    assert "SMALL_D_TILE_MS" in c.__all__
    assert c.sweep_seam_state() is None, "every seam unset = the module cache key every caller has had"
    for d in (32, 64, 128, 192):
        assert c._tile_m_d128(d) == 128


@pytest.mark.parametrize("d", [32, 64])
def test_the_seam_moves_the_height_the_slabs_the_block_rows_and_the_budgets_together(d, _pins):
    """One arithmetic (rule 8b'): the factory's TILE_M, the Q / O slabs, the bias block rows and the resident / ring budgets all read the
    pinned height -- at 64 rows the pair's resident block holds 1792 / 1152 kv columns at D 32 / 64 (the tile-height budget rows of the small-tile notes), the
    one-batch-entry's 896 / 512, and the 64-wide tile's one-batch-entry item gains the streamed ring it has none of at 128 rows."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    make = c.make_cfg_d32 if d == 32 else c.make_cfg_d64
    _pins(tile_m=None)
    base, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=d, bias_mode=1, n_chunk=4))
    assert base.TILE_M == 128 and base.BIAS_BLOCK_ROWS == 128
    assert (c.sm107_bias_resident_s_kv_max(d, 1, pair=True), c.sm107_bias_resident_s_kv_max(d, 1, pair=False)) == ((768, 384) if d == 32 else (384, 128))
    _pins(tile_m=64)
    assert c.sweep_seam_state() == ("sm107_sweep_seams", None, None, None, None, 64)
    cfg, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=d, bias_mode=1, n_chunk=4))
    assert cfg.TILE_M == 64 and cfg.BIAS_BLOCK_ROWS == 64 and cfg.TILE_K == cfg.TILE_O == d
    assert (
        cfg.TILES_Q * cfg.TILE_M * (cfg.TILE_K + cfg.TILE_O) * cfg.BPE == (base.TILES_Q * base.TILE_M * (base.TILE_K + base.TILE_O) * base.BPE) // 2
    ), "the Q / O slabs halve"
    operands = c.sdpa_smem_bytes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.TILES_Q, cfg.STAGES_KV, cfg.CTA_MMA, cfg.BPE, cfg.BPE_O)
    spare = c.SMEM_USABLE_BYTES - c._SMEM_FIXED_OVERHEAD - operands
    assert cfg.BIAS_RESIDENT_TILES == spare // (64 * cfg.TILE_N * cfg.BPE)
    assert (c.sm107_bias_resident_s_kv_max(d, 1, pair=True), c.sm107_bias_resident_s_kv_max(d, 1, pair=False)) == ((1792, 896) if d == 32 else (1152, 512))
    one, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=d, bias_mode=1))
    assert one.TILE_M == 64 and one.BIAS_BLOCK_ROWS == 128, "the one-batch-entry item's block is both sub-tiles' TILES_Q x TILE_M rows"
    if d == 64:
        assert c.sm107_bias_ring_stages(64, 1, pair=False) == 3, "the 64-wide one-batch-entry item streams at 64 rows (no ring fits at 128)"
    plain, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=d))
    assert plain.TILE_M == 64 and plain.BIAS_BLOCK_ROWS == 0


@pytest.mark.parametrize("bad", [32, 96, 256])
def test_a_height_outside_the_set_raises_at_the_seam_and_at_the_validator(bad, _pins):
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    _pins(tile_m=bad)
    with pytest.raises(ValueError, match="validated at"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=32))
    _pins(tile_m=None)
    cfg, _ = c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=32))
    with pytest.raises(ValueError, match="TILE_M must be one of"):
        c._validate_head_dim_tile(dataclasses.replace(cfg, TILE_M=bad), "probe")


def test_the_64_row_tile_declines_cga2_thd_split_packgqa_gate_and_the_quantized_dtypes(_pins):
    """Typed declines, each message naming the coupling: cga2 (a cta_group::2 MMA with a 64-row per-CTA tile has another datapath layout),
    THD / split / PackGQA / gate (the small-tile rows' own gates), fp8 (the quantized siblings keep the 128-row tile)."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    _pins(tile_m=64)
    with pytest.raises(ValueError, match="cga1 only"):
        c.make_cfg_d64(TemplateParams(cta_mma=2, dtype_qkv=2, d_flavor=64))
    _pins(tile_m=None)
    cfg, _ = c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=32))
    for repl, pat in (
        (dict(TILE_M=64, CTA_MMA=2, CGA_M=2), "cga1 only"),
        (dict(TILE_M=64, THD_VARLEN=1), "dense, unsplit"),
        (dict(TILE_M=64, SPLIT_KV=2), "dense, unsplit"),
        (dict(TILE_M=64, PACK_GQA=1), "dense, unsplit"),
        (dict(TILE_M=64, EPILOGUE_GATE=1), "dense, unsplit"),
        (dict(TILE_M=64, DTYPE_QKV=0), "f16/bf16 32- / 64-wide tiles only"),
        (dict(TILE_M=64, TILE_K=128, TILE_O=128, Q_SWZ_BYTES=128, K_SWZ_BYTES=128, V_SWZ_BYTES=128, O_SWZ_BYTES=128), "f16/bf16 32- / 64-wide tiles only"),
    ):
        with pytest.raises(ValueError, match=pat):
            c._validate_head_dim_tile(dataclasses.replace(cfg, **repl), "probe")


def test_the_d128_and_quantized_factories_ignore_the_seam(_pins):
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    _pins(tile_m=64)
    for make, kw in (
        (c.make_cfg_d128, dict(cta_mma=2, dtype_qkv=2)),
        (c.make_cfg_d128, dict(cta_mma=1, dtype_qkv=2)),
        (c.make_cfg_d128_mxfp8, dict(cta_mma=2, dtype_qkv=0)),
    ):
        cfg, _ = make(TemplateParams(d_flavor=128, **kw))
        assert cfg.TILE_M == 128, (make.__name__, kw)
    assert c._tile_m_d128(128) == 128 and c._tile_m_d128(192) == 128


def test_the_tmem_layout_stats_width_is_a_parameter_with_the_record_default(_pins):
    """The stats slot: 8 columns per sub-tile keeps STATS_OFF at 544 (every committed record); the quad-fragment form's 24 columns move it to
    512 with the O slots still ending before it at every served width; an invalid width raises."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    for tile_m in (None, 64):
        _pins(tile_m=tile_m)
        for d, make in ((32, c.make_cfg_d32), (64, c.make_cfg_d64)):
            cfg, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=d))
            lay = c.d128_family_tmem_layout(cfg)
            assert (lay.STATS_OFF, lay.STATS_COLS_PER_SUBTILE) == (544, 8)
            assert lay.S_OFF == (0, 128) and lay.P_OFF == (64, 192) and lay.O_OFF == (256, 256 + d)
            wide = c.d128_family_tmem_layout(cfg, stats_cols_per_subtile=24)
            assert (wide.STATS_OFF, wide.STATS_COLS_PER_SUBTILE) == (512, 24) and wide.O_OFF[-1] + cfg.TILE_O <= wide.STATS_OFF
    with pytest.raises(ValueError, match="multiple of 8"):
        c.d128_family_tmem_layout(cfg, stats_cols_per_subtile=12)


def test_the_seam_state_separates_module_cache_entries(_pins):
    """The template-module cache keys on (path, TemplateParams[, seam state]): the same record at TILE_M 128 and at 64 must load TWO modules
    with two source digests -- without the extra key the second request would silently return the first rendering (the module cache
    predates every seam)."""
    from cudnn.frost import template_loader
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    params = TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=32)
    _pins(tile_m=None)
    m128 = _load_sm100_kernel_module((32, 32), params, rubin=True)
    _pins(tile_m=64)
    m64 = _load_sm100_kernel_module((32, 32), params, rubin=True)
    assert m128 is not m64 and m128.CFG.TILE_M == 128 and m64.CFG.TILE_M == 64
    assert m128.FROST_SOURCE_DIGEST != m64.FROST_SOURCE_DIGEST
    assert m64.CGA_TILE_M == 128 and m128.CGA_TILE_M == 256
    assert m64.LAYOUT.STATS_OFF == m128.LAYOUT.STATS_OFF == 544
    _pins(tile_m=None)
    assert _load_sm100_kernel_module((32, 32), params, rubin=True) is m128, "the seam unset = the cached 128-row module"
    assert any(len(k) == 3 for k in template_loader._MODULES), "the 64-row module is keyed with the seam state"


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
from cudnn.sdpa.fwd import config_sm107 as _c
_c._TILE_M_OVERRIDE_FOR_SWEEPS = params_kw.pop("__tile_m", None)
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("TILE_M", mod.CFG.TILE_M)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("BIAS_BLOCK_ROWS", mod.CFG.BIAS_BLOCK_ROWS)
print("BIAS_RESIDENT_TILES", mod.CFG.BIAS_RESIDENT_TILES)
print("CGA_TILE_M", mod.CGA_TILE_M)
print("STATS_OFF", mod.LAYOUT.STATS_OFF)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "CGAERRBAR": ("CGAERRBAR",),
    "UTMALDG": ("UTMALDG",),
    "FMNMX3": ("FMNMX3",),
    "BSSY": ("BSSY",),
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


_RECORDS = {
    **{arm: ("md5_fwd_d128_file_bias_off_sm_107a.txt", p) for arm, p in _s13._RECORD_ARMS.items()},
    **{arm: ("md5_fwd_d128_file_bias_resident_sm_107a.txt", p) for arm, p in _s15._RESIDENT_RECORD_ARMS.items()},
}


@pytest.mark.parametrize("arm", list(_RECORDS))
def test_the_128_row_renderings_are_byte_identical_to_the_committed_records(tmp_path, arm):
    """With the seam unset every rendering of the two committed md5 records (the pre-bias arms and the resident pair arms) is byte-identical:
    the TILE_M axis, its validator, the seam-aware module key and the form-0 lane mapping add no instruction to the 128-row kernel."""
    record, params = _RECORDS[arm]
    dsl, want = _record(record)
    _skip_unless_record_dsl(dsl)
    probe = _probe(tmp_path, dict(params), f"m128_{arm}")
    assert probe.expect["TILE_M"] == 128 and probe.expect["CGA_TILE_M"] == 2 * 128 * params.get("cta_mma", 1)
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the record's {want[arm]} -- the 128-row rendering changed"


@pytest.mark.parametrize(
    "d, extra",
    [
        (32, {}),
        (32, {"dtype_qkv": 3, "window_right": 0}),
        (32, {"seq_kv_lens_present": True}),
        (32, {"bias_mode": 1, "n_chunk": 4}),
        (64, {}),
        (64, {"bias_mode": 1, "n_chunk": 4}),
    ],
    ids=["d32_dense", "d32_causal_f16", "d32_padded", "d32_pair_resident", "d64_dense", "d64_pair_resident"],
)
def test_the_64_row_renderings_differ_keep_the_split_and_spill_no_more(tmp_path, d, extra):
    """The 64-row renderings differ from their 128-row twins (m_dim 64, the warp-lane map, the masked stores), keep the per-role register split
    (USETMAXREG > 0), no cluster-scope drain (CGAERRBAR 0), the same TMA issue-site count (the tile height is not an issue site), the
    software row max where a mask is on, and spill no more than the 128-row rendering of the same arm; the bias block is 64 rows."""
    base_kw = {"__d": d, "cta_mma": 1, "dtype_qkv": 2, **extra}
    base = _probe(tmp_path, dict(base_kw), f"m128_{d}_{'_'.join(sorted(extra))}")
    probe = _probe(tmp_path, {**base_kw, "__tile_m": 64}, f"m64_{d}_{'_'.join(sorted(extra))}")
    assert base.expect["TILE_M"] == 128 and probe.expect["TILE_M"] == 64
    assert probe.expect["CGA_TILE_M"] == 128 and probe.expect["STATS_OFF"] == 544
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert probe.stats["UTMALDG"] == base.stats["UTMALDG"], (probe.stats, base.stats)
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} TILE_M 64 {sorted(extra)} vs 128: ")
    if extra.get("bias_mode"):
        assert probe.expect["BIAS_BLOCK_ROWS"] == 64 and probe.expect["BIAS_RESIDENT_TILES"] > base.expect["BIAS_RESIDENT_TILES"]


# --------------------------------------------------------------------------- tier 3: GPU (Rubin) -- the ring-wrap axis at 64 rows

# n_kv = 1 / 2 / 3 / 4 / 8 (S_kv 128 .. 1024 at TILE_N 128) with B x H > 1; a causal + padded row; the pair item (resident at 64 rows up to
# 1792 columns); the one-batch-entry bias item; a zero-length KV batch; the 64-wide tile; S_q not a multiple of 64 (a tail q tile).
_CASES = {
    "plain_nkv1_128": dict(b=4, h=2, h_kv=2, s_q=128, s_kv=128),
    "plain_nkv2_256": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256),
    "plain_nkv3_384_causal": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, causal=True),
    "plain_nkv4_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512),
    "plain_nkv8_1024": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024),
    "plain_640_padded_zero_batch": dict(b=4, h=2, h_kv=2, s_q=640, s_kv=640, kv_lens=[640, 0, 300, 640]),
    "plain_tail_q_200": dict(b=2, h=2, h_kv=2, s_q=200, s_kv=200, kv_lens=[200, 136]),
    "pair_512_dense": dict(b=4, h=2, h_kv=2, s_q=512, s_kv=512, n_chunk=4),
    "pair_1024_causal_padded": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=2, causal=True, kv_lens=[1024, 0, 700, 1024]),
    "one_batch_entry_per_batch_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, bias="b"),
    "d64_plain_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, d=64),
    "d64_pair_640_causal": dict(b=4, h=2, h_kv=2, s_q=640, s_kv=640, d=64, n_chunk=2, causal=True),
}


def _has_bias(c):
    return "bias" in c or "mode" in c or "n_chunk" in c


def _case_inputs(case, dtype, *, kv_mask_p=None):
    """The shared builder always draws a bias; a PLAIN case runs the kernel without one and its oracle sees a zero bias (s + 0 == s exactly),
    so the plain kernel is checked against the plain oracle."""
    c = _CASES[case]
    q, k, v, kw, ref = _s13._case_inputs(case, dtype, kv_mask_p=kv_mask_p, table=_CASES)
    if not _has_bias(c):
        ref = _s13._ref_of(c, q, k, v, {**kw, "bias": torch.zeros_like(kw["bias"])})
        kw = {**kw, "bias": None}
    return q, k, v, kw, ref


def _run(q, k, v, kw, c, _pins, *, tile_m):
    _pins(tile_m=tile_m, mode=c.get("mode"), n_chunk=c.get("n_chunk"))
    return _s13._run(q, k, v, **kw)


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES))
def test_the_64_row_tile_matches_the_fp64_oracle(case, _pins):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    out = _run(q, k, v, kw, c, _pins, tile_m=64)
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES))
def test_the_64_row_reference_form_is_bitwise_the_128_row_tile(case, _pins):
    """Form 0's live lanes run the 128-row tile's per-row arithmetic on the same operands in the same order (the M = 64 MMA reduces K in the
    same order per element; the softmax and the correction are per-lane), so O and Stats are BITWISE the 128-row kernel's -- a pin the
    quad-fragment form will NOT keep (its butterfly sums in another order; max stays bitwise)."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    narrow = _run(q, k, v, kw, c, _pins, tile_m=64)
    wide = _run(q, k, v, kw, c, _pins, tile_m=None)
    _s13._check(wide, ref)
    assert torch.equal(narrow["o"], wide["o"]) and torch.equal(narrow["stats"], wide["stats"]), f"{case}: TILE_M 64 form 0 differs from 128"


@requires_rubin
@pytest.mark.parametrize("case", ["plain_nkv3_384_causal", "pair_1024_causal_padded"])
def test_the_64_row_tile_composes_with_the_kv_mask(case, _pins):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    out = _run(q, k, v, kw, c, _pins, tile_m=64)
    _s13._check(out, ref)
    wide = _run(q, k, v, kw, c, _pins, tile_m=None)
    assert torch.equal(out["o"], wide["o"]) and torch.equal(out["stats"], wide["stats"])


@requires_rubin
@pytest.mark.parametrize("case", ["plain_nkv2_256", "pair_512_dense"])
def test_the_64_row_tile_matches_the_fp64_oracle_fp16(case, _pins):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.float16)
    out = _run(q, k, v, kw, c, _pins, tile_m=64)
    _s13._check(out, ref)
