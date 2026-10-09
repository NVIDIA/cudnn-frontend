# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K / V ring depth (``Cfg.STAGES_KV``) of the Rubin 32- / 64-wide f16/bf16 prefill tiles as a measured lever.

The small-tile body (``sm107/prefill_d128_f16.py`` at ``TemplateParams.d_flavor`` 32 / 64) is depth-generic over its K / V ring: no
accumulator parity is indexed from it and ``PipelineState.advance`` runs index and phase over any depth on the TMA-LDG and the MMA side
alike (the pair item's two tiles per kv step included), so the depth is a pure LATENCY knob: O and Stats are BITWISE identical at 2 / 3 / 4
stages, and a shallower ring frees SMEM the bias budgets read (``sm107_bias_resident_tiles`` / ``sm107_bias_ring_stages`` call the same
``_stages_kv_d128`` the factory calls -- one arithmetic).  The depth is pinned through the private, process-local seam
``config_sm107._STAGES_KV_OVERRIDE_FOR_SWEEPS`` (2 / 3 / 4 = ``SMALL_D_STAGES_KV``; the A/B's arm, never a plan path), the derived depth
stays 4.  Tiers: CPU contract (the seam moves the depth and the budgets together, raises outside the validated set, every existing
TemplateParams record renders at depth 4), host rendering pins (sm_107a trace-compile: the depth-4 renderings are byte-identical to the
committed md5 records, the 2- / 3-stage renderings differ, keep the register split, no cluster drain, spill no more than depth 4),
GPU (Rubin): the fp64 oracle at 2 / 3 stages on the ring-wrap axis (n_kv = depth, depth + 1, two wraps), bitwise 2 == 3 == 4.
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
    """Pin the K / V ring depth, the bias FORM, the bias ring depth and the chunk length a knob-less plan gets (the config module's private,
    process-local seams); ``None`` = the pick."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(stages_kv=None, mode=None, stages=None, n_chunk=None):
        monkeypatch.setattr(c, "_STAGES_KV_OVERRIDE_FOR_SWEEPS", stages_kv)
        monkeypatch.setattr(c, "_BIAS_MODE_OVERRIDE_FOR_SWEEPS", mode)
        monkeypatch.setattr(c, "_BIAS_STAGES_OVERRIDE_FOR_SWEEPS", stages)
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", n_chunk)

    return _set


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_the_seam_is_private_and_the_derived_depth_is_four():
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c._STAGES_KV_OVERRIDE_FOR_SWEEPS is None, "the seam is a per-test / per-sweep pin, never a module default"
    assert c.SMALL_D_STAGES_KV == frozenset({2, 3, 4})
    assert "SMALL_D_STAGES_KV" in c.__all__
    for d in (32, 64):
        for cta in sorted(c.SM107_SMALL_D_CGAS[(d, d)]):
            assert c._stages_kv_d128(2, cta, mxfp8=False, tile_k=d) == 4


@pytest.mark.parametrize("stages_kv", [2, 3, 4])
@pytest.mark.parametrize("d", [32, 64])
def test_the_seam_moves_the_depth_and_the_bias_budgets_together(d, stages_kv, _pins):
    """One arithmetic (rule 8b'): the factory's STAGES_KV, the SMEM footprint and the bias budgets all read the pinned depth."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    _pins(stages_kv=stages_kv)
    make = c.make_cfg_d32 if d == 32 else c.make_cfg_d64
    cfg, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=d))
    assert cfg.STAGES_KV == stages_kv
    operands = c.sdpa_smem_bytes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.TILES_Q, cfg.STAGES_KV, cfg.CTA_MMA, cfg.BPE, cfg.BPE_O)
    spare = c.SMEM_USABLE_BYTES - c._SMEM_FIXED_OVERHEAD - operands
    pair_tile = c.sm107_bias_block_rows(2, 128, pair=True) * cfg.TILE_N * cfg.BPE
    assert c.sm107_bias_resident_tiles(d, 1, pair=True) == spare // pair_tile
    assert c.sm107_bias_resident_s_kv_max(d, 1, pair=True) == (spare // pair_tile) * cfg.TILE_N
    # the resident pair rendering at this depth carries exactly that many tiles
    cfg_b, _ = make(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=d, bias_mode=1, n_chunk=4))
    assert cfg_b.STAGES_KV == stages_kv and cfg_b.BIAS_RESIDENT_TILES == spare // pair_tile


def test_the_plans_896_and_512_thresholds_follow_from_the_shallower_ring(_pins):
    """The plan's 'levers when it does not fit' row: at D 32 a 2- / 3-stage ring lifts the pair item's resident S_kv from 768 to 896; at
    D 64 a 2-stage ring lifts it from 384 to 640 (3 stages: 512) and gives the one-batch-entry item a 2-stage bias ring it has none of at 4."""
    from cudnn.sdpa.fwd import config_sm107 as c

    _pins(stages_kv=None)
    assert (c.sm107_bias_resident_s_kv_max(32, 1, pair=True), c.sm107_bias_resident_s_kv_max(64, 1, pair=True)) == (768, 384)
    assert c.sm107_bias_ring_stages(64, 1, pair=False) == 0
    _pins(stages_kv=3)
    assert (c.sm107_bias_resident_s_kv_max(32, 1, pair=True), c.sm107_bias_resident_s_kv_max(64, 1, pair=True)) == (896, 512)
    _pins(stages_kv=2)
    assert (c.sm107_bias_resident_s_kv_max(32, 1, pair=True), c.sm107_bias_resident_s_kv_max(64, 1, pair=True)) == (896, 640)
    assert c.sm107_bias_ring_stages(64, 1, pair=False) == 2


@pytest.mark.parametrize("bad", [1, 5])
def test_a_depth_outside_the_validated_set_raises_at_the_seam_and_at_the_validator(bad, _pins):
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    _pins(stages_kv=bad)
    with pytest.raises(ValueError, match="validated at"):
        c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=32))
    _pins(stages_kv=None)
    cfg, _ = c.make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2, d_flavor=32))
    with pytest.raises(ValueError, match="validated at STAGES_KV"):
        c._validate_head_dim_tile(dataclasses.replace(cfg, STAGES_KV=bad), "probe")


def test_the_d128_and_quantized_depths_are_untouched(_pins):
    """The seam is the small tiles' alone: d128 f16 keeps 2 x CTA_MMA, d192 keeps 2, per-tensor FP8 d128 cga2 keeps 9."""
    from cudnn.sdpa.fwd import config_sm107 as c

    for pin in (None, 2):
        _pins(stages_kv=pin)
        assert c._stages_kv_d128(2, 2, mxfp8=False, tile_k=128) == 4 and c._stages_kv_d128(2, 1, mxfp8=False, tile_k=128) == 2
        assert c._stages_kv_d128(2, 2, mxfp8=False, tile_k=192) == 2
        assert c._stages_kv_d128(0, 2, mxfp8=False, tile_k=128) == 9


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
from cudnn.sdpa.fwd import config_sm107 as _c
_c._STAGES_KV_OVERRIDE_FOR_SWEEPS = params_kw.pop("__stages_kv", None)
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("STAGES_KV", mod.CFG.STAGES_KV)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("BIAS_RESIDENT_TILES", mod.CFG.BIAS_RESIDENT_TILES)
print("STAGES_BIAS", mod.CFG.STAGES_BIAS)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {"USETMAXREG": ("USETMAXREG",), "STL": (" STL",), "LDL": (" LDL",), "CGAERRBAR": ("CGAERRBAR",), "UTMALDG": ("UTMALDG",), "FMNMX3": ("FMNMX3",)}


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
def test_the_derived_depth_renders_byte_identically_to_the_committed_record(tmp_path, arm):
    """With the seam unset (None) every existing resident rendering is byte-identical to its committed md5 record: the lever adds no
    instruction to the depth-4 kernel (the seam and the predicate are config-side)."""
    dsl, want = _record("md5_fwd_d128_file_bias_resident_sm_107a.txt")
    _skip_unless_record_dsl(dsl)
    probe = _probe(tmp_path, dict(_s15._RESIDENT_RECORD_ARMS[arm]), f"kv4_{arm}")
    assert probe.expect["STAGES_KV"] == 4
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the record's {want[arm]} -- the depth-4 rendering changed"


@pytest.mark.parametrize(
    "d, extra, stages_kv",
    [
        (32, {}, 2),
        (32, {}, 3),
        (32, {"bias_mode": 1, "n_chunk": 4}, 2),
        (32, {"bias_mode": 2, "n_chunk": 4}, 3),
        (64, {}, 2),
        (64, {"bias_mode": 1, "n_chunk": 4}, 3),
    ],
    ids=["d32_plain_2", "d32_plain_3", "d32_pair_resident_2", "d32_pair_streamed_3", "d64_plain_2", "d64_pair_resident_3"],
)
def test_a_shallower_ring_renders_a_different_kernel_with_the_same_split_and_no_new_spills(tmp_path, d, extra, stages_kv):
    """The 2- / 3-stage renderings differ from depth 4 (the ring extent and the prologue's pre-arm count move), keep the per-role register
    split (USETMAXREG > 0), no cluster-scope drain (CGAERRBAR 0), the same TMA issue-site count (the ring depth is not an issue site) and
    spill no more than the depth-4 rendering of the same arm."""
    base = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, **extra}, f"kv4_{d}_{'_'.join(sorted(extra))}")
    probe = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, "__stages_kv": stages_kv, **extra}, f"kv{stages_kv}_{d}_{'_'.join(sorted(extra))}")
    assert base.expect["STAGES_KV"] == 4 and probe.expect["STAGES_KV"] == stages_kv
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert probe.stats["UTMALDG"] == base.stats["UTMALDG"], (probe.stats, base.stats)
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} STAGES_KV {stages_kv} {sorted(extra)} vs depth 4: ")
    if extra.get("bias_mode") == 1:
        assert probe.expect["BIAS_RESIDENT_TILES"] > base.expect["BIAS_RESIDENT_TILES"], "a shallower ring frees SMEM for the resident block"


# --------------------------------------------------------------------------- tier 3: GPU (Rubin) -- the ring-wrap axis at every depth

# n_kv tiles = depth, depth + 1, two wraps (at TILE_N 128: S_kv 256 / 384 / 512 / 640 / 1024); the pair item, the one-batch-entry item,
# a causal + padded row, the 64-wide tile, a kv_mask row (TILE_N 64 pick: n_kv doubles), the bias ring.
_CASES = {
    "plain_256": dict(b=4, h=2, h_kv=2, s_q=256, s_kv=256),
    "plain_384_causal": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, causal=True),
    "plain_640_padded": dict(b=4, h=2, h_kv=2, s_q=640, s_kv=640, kv_lens=[640, 0, 300, 640]),
    "pair_512_dense": dict(b=4, h=2, h_kv=2, s_q=512, s_kv=512, n_chunk=4),
    "pair_1024_causal_padded": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=2, causal=True, kv_lens=[1024, 0, 700, 1024]),
    "one_batch_entry_per_batch_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, bias="b"),
    "one_batch_entry_ring_1024": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024, mode=2),
    "d64_plain_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, d=64),
    "d64_pair_640_causal": dict(b=4, h=2, h_kv=2, s_q=640, s_kv=640, d=64, n_chunk=2, causal=True),
}


def _has_bias(c):
    return "bias" in c or "mode" in c or "n_chunk" in c


def _case_inputs(case, dtype, *, kv_mask_p=None):
    """The shared builder always draws a bias; a PLAIN case runs the kernel without one and its oracle sees a zero bias (s + 0 == s exactly,
    the identity pin of the streamed module), so the plain kernel is checked against the plain oracle."""
    c = _CASES[case]
    q, k, v, kw, ref = _s13._case_inputs(case, dtype, kv_mask_p=kv_mask_p, table=_CASES)
    if not _has_bias(c):
        ref = _s13._ref_of(c, q, k, v, {**kw, "bias": torch.zeros_like(kw["bias"])})
        kw = {**kw, "bias": None}
    return q, k, v, kw, ref


def _run(q, k, v, kw, c, _pins, *, stages_kv):
    _pins(stages_kv=stages_kv, mode=c.get("mode"), n_chunk=c.get("n_chunk"))
    return _s13._run(q, k, v, **kw)


@requires_rubin
@pytest.mark.parametrize("stages_kv", [2, 3])
@pytest.mark.parametrize("case", list(_CASES))
def test_a_shallower_ring_matches_the_fp64_oracle_and_is_bitwise_the_derived_depth(case, stages_kv, _pins):
    """At 2 / 3 stages the kernel matches the fp64 oracle over the ring-wrap axis and is BITWISE the depth-4 kernel on O and Stats (a pure
    latency knob: the same tiles in the same order through the same instructions; only the SMEM extent and the pre-arm count differ)."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    shallow = _run(q, k, v, kw, c, _pins, stages_kv=stages_kv)
    _s13._check(shallow, ref)
    deep = _run(q, k, v, kw, c, _pins, stages_kv=None)
    assert torch.equal(shallow["o"], deep["o"]) and torch.equal(shallow["stats"], deep["stats"]), f"{case}: STAGES_KV {stages_kv} differs from 4"


@requires_rubin
@pytest.mark.parametrize("case", ["plain_384_causal", "pair_1024_causal_padded"])
def test_a_shallower_ring_composes_with_the_kv_mask(case, _pins):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    out = _run(q, k, v, kw, c, _pins, stages_kv=2)
    _s13._check(out, ref)
    deep = _run(q, k, v, kw, c, _pins, stages_kv=None)
    assert torch.equal(out["o"], deep["o"]) and torch.equal(out["stats"], deep["stats"])
