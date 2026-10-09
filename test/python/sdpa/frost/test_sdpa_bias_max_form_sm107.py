# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ROW-MAX FORM of the bias / kv_mask softmax body on the Rubin (SM107) f16/bf16 small head-dim tiles --
``TemplateParams.bias_max_form`` (``CFG.BIAS_MAX_FORM``) of ``sm107/prefill_d128_f16.py``: 0 = the SOFTWARE row max over the biased,
-inf-masked scores (the body as built), 1 = the HARDWARE UPPER BOUND -- the plain body's fused ``tcgen05.ld.red...max`` over the RAW
scores plus the thread's per-ROW bias-tile max in the packed 16-bit domain (``max.bf16x2`` / ``max.f16x2`` over the words the FFMA
reads), the online softmax on ``max_k S + max_k bias >= max_k (S + bias)``.  Any m >= the true row max is valid for the online softmax
(P <= 2^RESCALE_THRESHOLD as before; the LSE ``m + log(sum)`` and O = acc / sum are shift-invariant); the per-row bound's slack is the row's
RAW score range inside the tile -- tens of log2 units against the fp32 flush at -126 -- so a column is lost only when its true weight is
below 2^-(126 - slack): invisible in a bf16 O and in the fp32 LSE.  The bit-word kv_mask is ignored by the bound (masked columns can
only lower the true max); the additive kv_mask vector has no per-tile max to bound and is a typed decline with the form.

Three tiers: (1) CPU contract -- the appended TemplateParams / Cfg fields, the config factory's accept / raise table, the pick and its
sweep seam, the adapter's typed declines; (2) host rendering pins (sm_107a trace-compile) -- the SOFTWARE renderings of every bias /
kv_mask arm byte-identical to the committed record (the default form is unchanged), the hardware renderings with NO software max tree
(FMNMX3 0), the packed max present (HMNMX2), the fused loads (LDTM count unchanged), the register split kept, no cluster drain and no
new spills; (3) GPU cells on a Rubin device -- the hardware form against the fp64 oracle AND against the software form over the bias
module's degenerate-input sweep, the three bias ranges of the numerics protocol (N(0, 1), a -1e4 masked fold, a 10 000-scale bf16 bias)
with the worst-case slack |m_upper - m_true| documented per cell, the kv_mask fully masked row exactly 0 / -inf, the ring-wrap axis on
the kv_mask arm; the MEASURED per-form default (``config_sm107.sm107_bias_max_form_default``: the hardware bound on the kv_mask-only form,
the software tree with a bias) evidenced on the executed plan's record, and the default kv_mask-only plan's identity property (a mask of
all True is bitwise the plain kernel).
"""

import dataclasses
import math
import re
from pathlib import Path

import pytest
import torch

import test_sdpa_bias_sm107 as _s13
from frost_test_utils import assert_no_new_spills, requires_dsl, requires_rubin, run_sass_probe, sass_probe_source

pytestmark = [pytest.mark.L0, requires_dsl]

_KERNEL = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "kernels" / "sm107" / "prefill_d128_f16.py"
_RENDERINGS = Path(__file__).resolve().parent / "renderings"
_RECORD = _RENDERINGS / "md5_fwd_d128_file_bias_software_sm_107a.txt"


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


@pytest.fixture
def _hw_form(monkeypatch):
    """Pin the hardware upper-bound form on every knob-less plan of the test (the sweep seam; reset by monkeypatch)."""
    from cudnn.sdpa.fwd import config_sm107 as c

    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", 1)
    yield


# --------------------------------------------------------------------------- tier 1: CPU contract


def _params(**kw):
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    kw.setdefault("dtype_qkv", 2)
    kw.setdefault("cta_mma", 1)
    return TemplateParams(**kw)


def test_template_params_and_cfg_are_appended():
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    names = [f.name for f in dataclasses.fields(TemplateParams)]
    # appended after the two tail / quiet forms; the third softmax warp group's count (softmax_wgs, test_sdpa_softmax_wgs_sm107.py) after it
    assert names[-4:] == ["padded_tail_form", "dead_subtile_quiet", "bias_max_form", "softmax_wgs"], names[-6:]
    assert TemplateParams().bias_max_form == 0
    assert c.BIAS_MAX_SOFTWARE == 0 and c.BIAS_MAX_HW_UPPER_BOUND == 1 and c.BIAS_MAX_FORMS == frozenset({0, 1})
    # the record field defaults to 0 whatever the arm: a record without the field renders the software tree byte-identically
    for kw in (dict(bias_mode=1), dict(kv_mask_form=1), dict(bias_mode=1, kv_mask_form=1, tile_n=64), {}):
        cfg, _ = c.make_cfg_d32(_params(**kw))
        assert cfg.BIAS_MAX_FORM == 0, kw
    # the MEASURED per-form default the adapter picks: the hardware bound on the kv_mask-only form (+6..+15 % on seven cells), the software
    # tree wherever a bias is present (-5..-12 % on nine cells) and on a plan without either arm
    assert c.sm107_bias_max_form_default(0, 1) == 1
    assert c.sm107_bias_max_form_default(1, 0) == 0 and c.sm107_bias_max_form_default(2, 0) == 0
    assert c.sm107_bias_max_form_default(1, 1) == 0 and c.sm107_bias_max_form_default(2, 1) == 0
    assert c.sm107_bias_max_form_default(0, 2) == 0 and c.sm107_bias_max_form_default(0, 3) == 0 and c.sm107_bias_max_form_default(0, 0) == 0


@pytest.mark.parametrize(
    "d, kw",
    [
        (32, dict(bias_mode=1)),
        (32, dict(bias_mode=2, bias_stages=2)),
        (32, dict(kv_mask_form=1)),
        (32, dict(bias_mode=1, kv_mask_form=1, n_chunk=4, tile_n=64)),
        (32, dict(bias_mode=1, kv_mask_form=1, seq_kv_lens_present=True)),
        (64, dict(bias_mode=1, kv_mask_form=1, seq_kv_lens_present=True, cta_mma=2)),
        (32, dict(bias_mode=1, dtype_qkv=3)),
    ],
    ids=["d32_bias_res", "d32_bias_ring", "d32_mask", "d32_both_pair_t64", "d32_both_padded", "d64_both_cga2", "d32_bias_f16"],
)
def test_config_factory_accepts_the_hardware_form_on_every_served_arm(d, kw):
    from cudnn.sdpa.fwd import config_sm107 as c

    cfg, _ = (c.make_cfg_d32 if d == 32 else c.make_cfg_d64)(_params(bias_max_form=1, **kw))
    assert cfg.BIAS_MAX_FORM == 1
    base, _ = (c.make_cfg_d32 if d == 32 else c.make_cfg_d64)(_params(**kw))
    # a pure instruction-stream axis: every other derived field of the config is the software form's
    assert dataclasses.replace(cfg, BIAS_MAX_FORM=0) == base


@pytest.mark.parametrize(
    "kw, needle",
    [
        (dict(bias_max_form=1), "neither arm"),
        (dict(bias_max_form=1, kv_mask_form=2), "additive kv_mask vector"),
        (dict(bias_max_form=1, kv_mask_form=3, tile_n=64), "additive kv_mask vector"),
        (dict(bias_max_form=1, bias_mode=1, kv_mask_form=2), "additive kv_mask vector"),
        (dict(bias_max_form=2, bias_mode=1), "must be 0"),
    ],
    ids=["no_arm", "maskadd_16", "maskadd_f32", "bias_with_maskadd", "form_2"],
)
def test_config_factory_declines_the_hardware_form_where_it_is_not_served(kw, needle):
    from cudnn.sdpa.fwd import config_sm107 as c

    with pytest.raises(ValueError, match=needle):
        c.make_cfg_d32(_params(**kw))


def test_pick_default_seam_and_request(monkeypatch):
    from cudnn.sdpa.fwd import config_sm107 as c

    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", None)
    # no request, no seam: the measured per-form default
    assert c.sm107_bias_max_form_pick(0, 1) == 1, "the kv_mask-only form runs the hardware bound by default"
    assert c.sm107_bias_max_form_pick(1, 0) == 0 and c.sm107_bias_max_form_pick(2, 1) == 0 and c.sm107_bias_max_form_pick(0, 0) == 0
    assert c.sm107_bias_max_form_pick(0, 2) == 0 and c.sm107_bias_max_form_pick(0, 3) == 0
    assert c.sm107_bias_max_form_pick(0, 0, request=1) == 1, "a REQUEST is handed through (the factory / the adapter decline it typed)"
    assert c.sm107_bias_max_form_pick(0, 1, request=0) == 0, "a request for the software tree on the kv_mask-only form is honoured"
    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", 0)
    assert c.sm107_bias_max_form_pick(0, 1) == 0, "the seam pins the software form on the kv_mask-only plan (the A/B's base arm)"
    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", 1)
    assert c.sm107_bias_max_form_pick(1, 0) == 1 and c.sm107_bias_max_form_pick(0, 1) == 1 and c.sm107_bias_max_form_pick(2, 1) == 1
    # the picked DEFAULT degrades where the bound is not served (an invalid record is never produced by a default)
    assert c.sm107_bias_max_form_pick(0, 2) == 0 and c.sm107_bias_max_form_pick(1, 3) == 0 and c.sm107_bias_max_form_pick(0, 0) == 0


def test_adapter_kwarg_is_appended_and_declines_typed():
    import inspect

    from cudnn.sdpa.fwd import api_dsl

    sig = inspect.signature(api_dsl.SdpaFwdDslSm100.__init__)
    assert list(sig.parameters)[-2:] == ["bias_max_form", "softmax_wgs"] and sig.parameters["bias_max_form"].default is None
    assert "bias_max_form=self._resident_arm_bias_max_form()" in Path(api_dsl.__file__).read_text()


def test_kernel_carries_the_form_as_one_const_expr_axis():
    src = _KERNEL.read_text()
    assert "_HW_UPPER_MAX = CFG.BIAS_MAX_FORM == BIAS_MAX_HW_UPPER_BOUND" in src
    body = src[src.index("def _softmax_kv_body_inf(") : src.index("def _softmax_warp_group(")]
    assert body.count("if cutlass.const_expr(_HW_UPPER_MAX):") == 2, "the load site and the max site fold on the ONE axis"
    assert "_tmem_load_max_reduction(s_addr_base + cutlass.Int32(c * CHUNK), num=CHUNK)" in body, "the plain body's fused primitive"
    assert "packed_row_max_words(" in body and "packed_pair_max_f32(" in body and "opaque_offset=True" in body
    assert "row_max_for_exp2(total_max)" in body, "the -inf form's 0-substitution is kept verbatim"
    assert "row_max_reduction(chunks_S[c])" in body, "the software tree stays the other arm"
    assert "if _HW_UPPER_MAX and _KV_MASK_ADD:" in src and "if _HW_UPPER_MAX and not _INF_MASK_ARM:" in src


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("KV_MASK_FORM", mod.CFG.KV_MASK_FORM)
print("BIAS_MAX_FORM", mod.CFG.BIAS_MAX_FORM)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "FMNMX3": ("FMNMX3",),
    "HMNMX2": ("HMNMX2",),
    "LDTM": (" LDTM",),
    "R2P": (" R2P ",),
    "CGAERRBAR": ("CGAERRBAR",),
    "FFMA2": (" FFMA2",),
    "MUFU_EX2": ("MUFU.EX2",),
}
# The bias / kv_mask arms whose SOFTWARE renderings the committed record pins (every served form: resident / ring, one-batch-entry /
# pair, 128 / 64 wide, bf16 / f16, cga1 / cga2, the additive vector).
_ARMS = {
    "d32_bias_res_one": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 1},
    "d32_bias_mask_padded": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 1, "kv_mask_form": 1, "seq_kv_lens_present": True},
    "d32_mask_only": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "kv_mask_form": 1},
    "d32_mask_only_t64": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "kv_mask_form": 1, "tile_n": 64},
    "d32_bias_pair_n4": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 1, "n_chunk": 4},
    "d32_bias_ring_st2": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, "bias_stages": 2},
    "d32_both_pair_n4_t64": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 1, "kv_mask_form": 1, "n_chunk": 4, "tile_n": 64},
    "d32_both_ring_st2_t64": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, "bias_stages": 2, "kv_mask_form": 1, "tile_n": 64},
    "d64_bias_mask_cga2_padded": {"__d": 64, "cta_mma": 2, "dtype_qkv": 2, "bias_mode": 1, "kv_mask_form": 1, "seq_kv_lens_present": True},
    "d32_maskadd_only": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "kv_mask_form": 2},
    "d32_bias_f16_res_one": {"__d": 32, "cta_mma": 1, "dtype_qkv": 3, "bias_mode": 1},
}
_HW_ARMS = [a for a in _ARMS if a != "d32_maskadd_only"]  # the additive vector keeps the software form (a typed decline with the hardware one)


def _record():
    assert _RECORD.is_file(), _RECORD
    dsl, want = None, {}
    for ln in _RECORD.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"fwd_d128_file sm_107a (\S+) rc=0 cubin_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


def _skip_unless_record_dsl():
    from cudnn.frost.buffers import cutedsl_state

    dsl, want = _record()
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: the cubin is a function of the DSL build, re-render the record")
    return want


def test_software_md5_record_is_committed_and_complete():
    dsl, want = _record()
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert set(want) == set(_ARMS), (sorted(want), sorted(_ARMS))


@pytest.mark.parametrize("arm", list(_ARMS))
def test_software_renderings_are_byte_identical_to_the_record(tmp_path, arm):
    """The RECORD default is the software max: with ``bias_max_form`` unset in the record every bias / kv_mask rendering traces
    byte-identically to the tree before the hardware form landed (the record was rendered from that tree).  The adapter's per-form pick
    (the hardware bound on a kv_mask-only PLAN) is a record FIELD it sets, evidenced on the executed plan below, not a change of the
    rendering a record without the field produces."""
    want = _skip_unless_record_dsl()
    probe = _probe(tmp_path, _ARMS[arm], f"sw_{arm}")
    assert probe.expect["BIAS_MAX_FORM"] == 0
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the record's {want[arm]} -- the software rendering changed"


@pytest.mark.parametrize("arm", _HW_ARMS)
def test_hardware_renderings_drop_the_tree_keep_the_split_and_spill_no_more(tmp_path, arm):
    """``bias_max_form = 1``: NO software max tree (FMNMX3 0 against the software form's 62 / 124), the packed bias max present on the
    bias arms (HMNMX2 = 63 per 128 columns per sub-tile body; 0 on the kv_mask-only arm, whose bound is the hardware max alone), the SAME
    LDTM count (the fused load replaces the plain one), the register split kept, no cluster drain, FFMA2 / MUFU unchanged and no new
    spills over the software rendering of the same arm (the early opaque second read of the non-pre-loaded chunk keeps the words out of
    the TMEM chunks' live range)."""
    want = _skip_unless_record_dsl()
    sw = _probe(tmp_path, _ARMS[arm], f"sw_{arm}")
    assert sw.cubin_md5 == want[arm]
    params = dict(_ARMS[arm])
    params["bias_max_form"] = 1
    hw = _probe(tmp_path, params, f"hw_{arm}")
    assert hw.expect["BIAS_MAX_FORM"] == 1 and hw.cubin_md5 != sw.cubin_md5
    assert hw.stats["FMNMX3"] == 0 and sw.stats["FMNMX3"] > 0, (hw.stats, sw.stats)
    assert hw.stats["LDTM"] == sw.stats["LDTM"], (hw.stats, sw.stats)
    assert hw.stats["USETMAXREG"] > 0 and hw.stats["CGAERRBAR"] == 0, hw.stats
    assert hw.stats["FFMA2"] == sw.stats["FFMA2"] and hw.stats["MUFU_EX2"] == sw.stats["MUFU_EX2"], (hw.stats, sw.stats)
    if _ARMS[arm].get("bias_mode"):
        assert hw.stats["HMNMX2"] == 2 * (_ARMS[arm].get("tile_n", 128) // 2 - 1), hw.stats
    else:
        assert hw.stats["HMNMX2"] == 0, hw.stats
    if _ARMS[arm].get("kv_mask_form") or _ARMS[arm].get("seq_kv_lens_present"):
        assert hw.stats["R2P"] > 0, hw.stats
    assert_no_new_spills(hw.stats, sw.stats, tag=f"{arm} hw vs sw: ")


# --------------------------------------------------------------------------- tier 3: GPU cells (a Rubin device)

# The hardware form against the software form: both are the same function up to the fp32 rounding of the exp2 arguments (a different m
# per step), so O agrees to a bf16 ulp or two and the LSE to a few fp32 ulps -- pinned HERE, tighter than the oracle tolerances.
_TOL_O_VS_SW = dict(atol=1.6e-2, rtol=1.6e-2)  # bf16: 8 mantissa bits
_TOL_LSE_VS_SW = dict(atol=2e-5, rtol=2e-5)

_SWEEP = [
    "dense_one_tile_bh1",
    "dense_nkv2",
    "dense_nkv3_384",
    "dense_tail_304",
    "causal_384",
    "causal_tail_376",
    "right_band_64_tail_304",
    "causal_swa128_padded_384",
    "causal_gqa_8_2",
    "padded_kv_zero_batch",
    "causal_sink_padded_q_trim",
    "bias_head_broadcast",
    "bias_per_batch",
    "bias_x10",
    "bias_x10_causal_384",
    "bias_neg1e9_key_fold",
    "d64_dense_128",
    "d64_dense_256_cga2",
    "d64_causal_256_cga2",
    "dense_one_tile_64",
    "tail_217_padded_stride_bias",
]


def _run_both(q, k, v, kw, monkeypatch):
    """The software and the hardware form on the SAME inputs (the seam flipped between the two builds)."""
    from cudnn.sdpa.fwd import config_sm107 as c

    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", 0)
    sw = _s13._run(q, k, v, **kw)
    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", 1)
    hw = _s13._run(q, k, v, **kw)
    return sw, hw


def _check_hw(sw, hw, ref):
    _s13._check(hw, ref)
    live = ~ref["empty"]
    torch.testing.assert_close(hw["o"][live].double(), sw["o"][live].double(), **_TOL_O_VS_SW)
    torch.testing.assert_close(hw["stats"].squeeze(-1)[live].double(), sw["stats"].squeeze(-1)[live].double(), **_TOL_LSE_VS_SW)
    assert torch.equal(hw["o"][ref["empty"]], sw["o"][ref["empty"]]) and torch.equal(
        hw["stats"].squeeze(-1)[ref["empty"]], sw["stats"].squeeze(-1)[ref["empty"]]
    )


def _slack(q, k, bias, scale, kv_mask=None, tile_n=128):
    """Host model of the bound: per (row, kv tile) ``m_upper - m_true`` in log2 units -- the hardware raw max + the per-row bias-tile max
    against the true max of the biased (and masked) scores.  Returns (max slack, the raw range it is bounded by)."""
    k_ref = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1) if k.shape[1] != q.shape[1] else k  # GQA: the shared KV head per Q head
    s = torch.matmul(q.double(), k_ref.double().transpose(-1, -2))  # (b, h, s_q, s_kv) raw
    b_ = (bias.double() if bias is not None else torch.zeros_like(s)).expand_as(s)
    worst, worst_range = 0.0, 0.0
    for t0 in range(0, s.shape[-1], tile_n):
        sr, br = s[..., t0 : t0 + tile_n], b_[..., t0 : t0 + tile_n]
        m_upper = sr.max(-1).values * scale + br.max(-1).values
        true = sr * scale + br
        if kv_mask is not None:
            true = true.masked_fill(~kv_mask.view(kv_mask.shape[0], 1, 1, -1)[..., t0 : t0 + tile_n], float("-inf"))
        m_true = true.max(-1).values
        live = torch.isfinite(m_true)
        if live.any():
            worst = max(worst, float(((m_upper - m_true)[live] * math.log2(math.e)).max()))
            worst_range = max(worst_range, float(((sr.max(-1).values - sr.min(-1).values) * scale)[live].max() * math.log2(math.e)))
    return worst, worst_range


@requires_rubin
@pytest.mark.parametrize("case", _SWEEP)
def test_hw_form_matches_the_oracle_and_the_software_form(case, monkeypatch):
    """The degenerate-input sweep of the bias module (bf16) under the hardware form: O and Stats against the fp64 oracle at the module's
    tolerances, keyless rows exactly 0 / -inf, two launches bitwise (inside ``_run``), AND against the software form at the tighter
    same-function tolerances; the worst-case slack of the bound is printed with the cell."""
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16)
    sw, hw = _run_both(q, k, v, kw, monkeypatch)
    _check_hw(sw, hw, ref)
    slack, raw_range = _slack(q, k, kw["bias"], kw["scale"], kw["kv_mask"])
    print(f"\n{case}: worst |m_upper - m_true| {slack:.3f} log2 units (raw range bound {raw_range:.3f})")
    assert slack <= raw_range + 1e-6 and slack < 60.0, (slack, raw_range)


@requires_rubin
@pytest.mark.parametrize("case", ["dense_nkv3_384", "causal_swa128_padded_384", "bias_x10"])
def test_hw_form_fp16(case, monkeypatch):
    q, k, v, kw, ref = _s13._case_inputs(case, torch.float16)
    sw, hw = _run_both(q, k, v, kw, monkeypatch)
    _check_hw(sw, hw, ref)


@requires_rubin
@pytest.mark.parametrize("case", ["dense_nkv3_384", "causal_384", "padded_kv_zero_batch", "bias_neg1e9_key_fold"])
def test_hw_form_with_the_kv_mask(case, monkeypatch):
    """The bias WITH the bit-word kv_mask (30 % masked): the bound ignores the mask -- a masked column can only lower the true max."""
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    sw, hw = _run_both(q, k, v, kw, monkeypatch)
    _check_hw(sw, hw, ref)
    slack, raw_range = _slack(q, k, kw["bias"], kw["scale"], kw["kv_mask"])
    print(f"\n{case}+kv_mask: worst |m_upper - m_true| {slack:.3f} log2 units (raw range bound {raw_range:.3f})")
    assert slack <= raw_range + 1e-6


@requires_rubin
@pytest.mark.parametrize("case", list(_s13._MASK_RING_CASES))
def test_hw_form_kv_mask_alone_across_the_ring_wrap(case, monkeypatch):
    """The kv_mask ALONE (no bias): the bound is the hardware raw max alone, over the ring-wrap axis (n_kv 4 / 5 / 8, a tail)."""
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_s13._MASK_RING_CASES)
    kw["bias"] = None
    ref = _s13._ref_of(_s13._MASK_RING_CASES[case], q, k, v, kw)
    sw, hw = _run_both(q, k, v, kw, monkeypatch)
    _check_hw(sw, hw, ref)


def _profile_bias(kind, bb, bh, s_q, s_kv, dtype):
    """The numerics protocol's bias RANGES: the -1e4 masked fold (a finite masked value, row-structured: 30 % of the keys), the
    10 000-scale bias (bf16 quantum 64 at that magnitude), and a per-ROW structured bias (one row all -1e4, one row all +1e4)."""
    if kind == "neg1e4_fold":
        keep = torch.rand(bb, 1, 1, s_kv, device="cuda") > 0.3
        return torch.where(keep, torch.zeros(bb, bh, s_q, s_kv, device="cuda"), torch.full((bb, bh, s_q, s_kv), -1e4, device="cuda")).to(dtype)
    if kind == "scale_1e4":
        return (torch.randn(bb, bh, s_q, s_kv, device="cuda") * 1e4).to(dtype)
    if kind == "row_structured":
        b = torch.randn(bb, bh, s_q, s_kv, device="cuda")
        b[..., 3, :] = -1e4
        b[..., 5, :] = 1e4
        b[..., 7, : s_kv // 2] = -1e4
        return b.to(dtype)
    raise ValueError(kind)


@requires_rubin
@pytest.mark.parametrize("kind", ["neg1e4_fold", "scale_1e4", "row_structured"])
@pytest.mark.parametrize("shape", [(2, 2, 384, 384, 32), (2, 2, 256, 256, 32), (1, 4, 1024, 1024, 32)], ids=["s384", "s256", "s1024_ring"])
def test_hw_form_on_the_protocol_bias_ranges(kind, shape, monkeypatch):
    """STEPS S16's numerics protocol: the bias ranges where a BLOCK-scalar bound would underflow whole rows (a -1e4 row under a 0 block max
    shrinks every exp by 2^-14 000); the per-ROW bound's slack stays the raw range -- no underflow-induced zero row (a live row's O is
    finite and its sum > 0 wherever the oracle's is), the oracle and the software form agree."""
    torch.manual_seed(7)
    b, h, s_q, s_kv, d = shape
    dtype = torch.bfloat16
    q, k, v = (_s13._bhsd(b, h, s_q, d, dtype) for _ in range(3))
    bias = _profile_bias(kind, 1, h, s_q, s_kv, dtype)
    scale = 1.0 / math.sqrt(d)
    kw = dict(scale=scale, bias=bias, kv_mask=None, sdpa_kwargs={}, seq_len_kv=None, seq_len_q=None, sink=None, cga=None)
    ref = _s13._ref(q, k, v, scale=scale, bias=bias)
    sw, hw = _run_both(q, k, v, kw, monkeypatch)
    _check_hw(sw, hw, ref)
    live = ~ref["empty"]
    assert torch.isfinite(hw["stats"].squeeze(-1)[live]).all(), "a live row's LSE under the bound must stay finite (no underflow-induced zero row)"
    slack, raw_range = _slack(q, k, bias, scale)
    print(f"\n{kind} {shape}: worst |m_upper - m_true| {slack:.3f} log2 units (raw range bound {raw_range:.3f})")
    assert slack <= raw_range + 1e-6 and slack < 60.0, (slack, raw_range)


def _spy_loaded_forms(monkeypatch):
    """Record ``params.bias_max_form`` of every kernel module the adapter loads (the record the EXECUTED plan carries; a memoised load
    still passes through the spy)."""
    from cudnn.sdpa.fwd import api_dsl, config_sm107 as c

    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", None)  # no seam: the adapter's own pick
    seen = []
    real = api_dsl._load_sm100_kernel_module

    def spy(flavor, params, *a, **kw):
        seen.append(int(getattr(params, "bias_max_form", -1)))
        return real(flavor, params, *a, **kw)

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", spy)
    return seen


@requires_rubin
@pytest.mark.parametrize("case", ["dense_nkv3_384", "causal_384", "padded_kv_zero_batch", "d64_dense_128"])
def test_the_default_pick_runs_the_hardware_form_on_the_kv_mask_only_plan_and_the_software_tree_with_a_bias(case, monkeypatch):
    """No request, no seam: a kv_mask-only plan loads its kernel with ``bias_max_form`` 1 (the measured default), a plan with a bias --
    alone or with the kv_mask -- with 0; both agree with the fp64 oracle at the module's tolerances (the pick is performance-only)."""
    q, k, v, kw, ref = _s13._case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    seen = _spy_loaded_forms(monkeypatch)
    out = _s13._run(q, k, v, **{**kw, "bias": None})
    _s13._check(out, _s13._ref_of(_s13._CASES[case], q, k, v, {**kw, "bias": None}))
    assert seen and set(seen) == {1}, f"kv_mask-only plan: loaded forms {seen}"
    del seen[:]
    out = _s13._run(q, k, v, **kw)
    _s13._check(out, ref)
    assert seen and set(seen) == {0}, f"bias + kv_mask plan: loaded forms {seen}"
    del seen[:]
    out = _s13._run(q, k, v, **{**kw, "kv_mask": None})
    _s13._check(out, _s13._ref_of(_s13._CASES[case], q, k, v, {**kw, "kv_mask": None}))
    assert seen and set(seen) == {0}, f"bias-only plan: loaded forms {seen}"


@requires_rubin
@pytest.mark.parametrize("dtype", _s13._DTYPES, ids=_s13._DTYPE_IDS)
@pytest.mark.parametrize("case", ["dense_nkv2", "dense_nkv3_384", "d64_dense_256_cga2"])
def test_default_kv_mask_all_true_is_bitwise_the_plain_kernel(case, dtype, monkeypatch):
    """The default kv_mask-only plan (the hardware bound) with a mask of all True is BITWISE the plain kernel on O and Stats on a DENSE
    cell: the bound is the fused raw max the plain body runs on, a keep word of all ones changes no value, and both tile the same width
    (two or more 128-wide kv tiles in every cell here: the kv_mask-only pick narrows to 64 at ONE tile only, where the plain kernel stays
    at 128).  Not on a band-masked cell: on a diagonal tile the plain body takes the EXACT max of the masked scores while the bound keeps
    the raw one -- the same function within rounding (the oracle sweep above covers causal), not bitwise."""
    from cudnn.sdpa.fwd import config_sm107 as c

    monkeypatch.setattr(c, "_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS", None)
    q, k, v, kw, _ = _s13._case_inputs(case, dtype)
    cc = _s13._CASES[case]
    plain = _s13._run(q, k, v, **{**kw, "bias": None, "kv_mask": None})
    ones = torch.ones(cc["b"], 1, 1, cc["s_kv"], device="cuda", dtype=torch.bool)
    seen = _spy_loaded_forms(monkeypatch)
    out = _s13._run(q, k, v, **{**kw, "bias": None, "kv_mask": ones})
    assert seen and set(seen) == {1}, seen
    assert torch.equal(out["o"], plain["o"]), "mask = all True under the hardware bound: O differs from the plain kernel"
    assert torch.equal(out["stats"], plain["stats"]), "mask = all True under the hardware bound: Stats differ from the plain kernel"


@requires_rubin
def test_hw_form_kv_mask_fully_masked_row_is_exactly_zero_and_minus_inf(_hw_form):
    """A kv_mask row with NO live key under the hardware form: the bound is finite (the raw max of a real S), every P is exp2(-inf) = 0,
    ``total_sum == 0`` and the epilogue's SELECT reads O = 0 / Stats = -inf EXACTLY (the -inf form is kept verbatim)."""
    torch.manual_seed(3)
    d = 32
    q, k, v = (_s13._bhsd(2, 2, 256, d, torch.bfloat16) for _ in range(3))
    kv_mask = torch.rand(2, 1, 1, 256, device="cuda") > 0.3
    kv_mask[1] = False
    out = _s13._run(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=kv_mask)
    assert (out["o"][1] == 0).all() and torch.isneginf(out["stats"][1]).all()
    ref = _s13._ref(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=kv_mask)
    _s13._check(out, ref)
