# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The score side outputs (``score_max`` / ``score_sum_exp``) of the FROST f16/bf16 d128 forward engines, SM100 and SM107.

The graph's two optional ports are fp32 ``(B, H_q, S_q, 1)`` row statistics in NATURAL units: ``score_max`` is the
softmax row max (``scale * max_j s_j``, the sink folded in as one more column) and ``score_sum_exp`` the fp32 row-sum of
``exp(s - max)`` -- the two operands the LSE is built from, so ``Stats == score_max + ln(score_sum_exp)`` to fp32 rounding
(``stats_use_log2`` leaves them untouched).  A keyless row reads ``-inf`` / ``0``, with a sink ``sink_logit`` / ``1``, the
same SELECT and the same OOB-row store guard as the LSE.  Served by the d128 f16/bf16 prefill kernels on dense, unsplit,
unpaged plans of the 2-CTA tile; every other selection is a typed decline (the engine row's ``mismatch`` and the standalone
adapter's ``check_support`` read ONE shape constant per arch line).

Three tiers: (1) CPU contract -- the rows' claims, the mismatch accept / reject table, the append-only signature pins and
the adapter's typed declines; (2) rendering pins -- a host trace-compile of the d128 kernel on sm_100a and sm_107a whose
no-score arm must stay byte-identical to the committed pre-edit cubin md5 records (``renderings/``) while the score arm
adds exactly the stores; (3) GPU cells on the Blackwell-line device present (sm100 or sm107 row): the fp64 oracle over the
degenerate-input matrix (the KV ring sweep n_kv = 1, 2, 3, 4, 5, 8 -- the sm107 KV ring depth 4 and the sm100 depth 2, each
with its depth + 1 -- all with B*H > 1; tail tiles; a zero-length KV batch; keyless bottom-right rows with and without a sink;
keyless rows past a left window under padding; GQA) in bf16 with the fp16 member on the dense base and on the ports' documented
degenerate contract (the keyless row -inf / 0, the sink sink_logit / 1), the base-2 Stats interaction, each output alone,
strided layouts, and the bitwise identity of O / Stats with the outputs off -- each once, on the 2-CTA tile the arm is wired on.
"""

import inspect
import math
import re
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from frost_test_utils import _SM, assert_no_new_spills, make_dense_stats, offers_engine, requires_blackwell, requires_dsl, run_sass_probe, sass_probe_source
from test_sdpa_fwd_decode_d256_sm107 import _DSL_FLOOR_WHY

pytestmark = [pytest.mark.L0, requires_dsl]

_ARCH = "sm107" if _SM == 107 else "sm100"
_ROW = f"sdpa_fwd_prefill_{_ARCH}"
_ROWS = {"sdpa_fwd_prefill_sm100": (10, 0), "sdpa_fwd_prefill_sm107": (10, 7)}


# --------------------------------------------------------------------------- tier 1: CPU contract


def _facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    base = dict(
        b=2,
        h_q=8,
        h_kv=8,
        s_q=512,
        s_kv=512,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.HALF,
        dtype_o=cudnn.data_type.HALF,
        device_cc=(10, 0),
        has_score_max=True,
        has_score_sum_exp=True,
    )
    base.update(kw)
    return ga.SdpaGraphFacts(**base)


def _caps(name):
    from cudnn.sdpa.fwd import engines

    return {s.name: s.capabilities for s in engines.ENGINE_SPECS}[name]


def test_rows_claim_exactly_the_wired_flavors():
    """The two f16/bf16 rows claim the outputs through the config constants (rule 8b': ONE definition for the row and the
    adapter); every other forward row declines them (quantized kernels, SM80 / SM90 / SM120 carry no store)."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm100 import SM100_SCORE_STATS_SHAPES
    from cudnn.sdpa.fwd.config_sm107 import SM107_SCORE_STATS_SHAPES

    assert SM100_SCORE_STATS_SHAPES == frozenset({(128, 128)}) and SM107_SCORE_STATS_SHAPES == frozenset({(128, 128)})
    for name, shapes in (("sdpa_fwd_prefill_sm100", SM100_SCORE_STATS_SHAPES), ("sdpa_fwd_prefill_sm107", SM107_SCORE_STATS_SHAPES)):
        caps = _caps(name)
        assert caps.score_max and caps.score_sum_exp, name
        assert caps.score_stats_d_shapes is shapes, name
    for spec in engines.ENGINE_SPECS:
        if spec.name not in _ROWS:
            assert not spec.capabilities.score_max and not spec.capabilities.score_sum_exp, spec.name


@pytest.mark.parametrize("row", list(_ROWS))
def test_mismatch_accepts_and_rejects(row):
    """Accept: dense d128 f16, each output alone, with Stats, with base-2 Stats, without Stats.  Reject (typed reasons):
    THD, paged KV, an unwired flavor, split_kv > 1 and the cga=1 tile."""
    if row == "sdpa_fwd_prefill_sm107" and _DSL_FLOOR_WHY:
        pytest.skip(_DSL_FLOOR_WHY)  # Rule 7: below the sm_107a DSL floor the Rubin row declines typed before any claim
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps, cc = _caps(row), _ROWS[row]

    def reason(knobs=None, **kw):
        return engines.mismatch(caps, _facts(device_cc=cc, **kw), knobs)

    assert reason() is None
    assert reason(has_score_sum_exp=False) is None
    assert reason(has_score_max=False) is None
    assert reason(wants_stats=True) is None
    assert reason(wants_stats=True, has_stats_log2=True) is None, "base-2 Stats leave the natural-unit score outputs untouched"
    assert reason(causal=True, right_bound=0) is None
    assert reason(padded=True) is None
    assert reason(has_sink=True) is None
    assert "dense graphs only" in reason(thd=True, packed_layout=True)
    assert "paged-KV" in reason(has_paged_kv=True, padded=True, page_size=128)
    assert "wired on the d128" in reason(d_qk=256, d_v=256)
    if row == "sdpa_fwd_prefill_sm100":
        # The native d64 flavor is the same file at d_flavor=64 but is not validated with the outputs: not claimed.
        assert "wired on the d128" in reason(d_qk=64, d_v=64)
        assert "wired on the d128" in reason(d_qk=24, d_v=24), "Blackwell's smallest flavor is its native d64"
    else:
        # RE-INVERTED 2026-10-08: on Rubin a d64 / d24 graph WITH the outputs rides the d128 envelope, i.e. the kernel that
        # carries the store, exactly as before the small head-dim tiles -- the tiles carry the dense leg only
        # (engines.Capabilities.dense_only_d_shapes), so a graph asking for the ports is selected past them.
        for d in (64, 24):
            assert reason(d_qk=d, d_v=d) is None, (d, reason(d_qk=d, d_v=d))
            assert engines._selected_d_shape(caps, _facts(device_cc=cc, d_qk=d, d_v=d)) == (128, 128), d
            assert engines._selected_d_shape(caps, _facts(device_cc=cc, d_qk=d, d_v=d, has_score_max=False, has_score_sum_exp=False)) != (128, 128), d
    assert "split_kv > 1" in reason(SdpaFwdKnobs(split_kv=2))
    assert "cga" in reason(SdpaFwdKnobs(cga=1))
    assert reason(SdpaFwdKnobs(cga=2, split_kv=1)) is None
    # The knob DOMAIN narrows to the prefill tile with the ports requested, so the heuristics never propose the
    # cga1 decode tile for a decode-shaped graph (S_q <= 128) and leave the row without a plan.
    assert engines.effective_cgas(caps, _facts(device_cc=cc, s_q=128)) == frozenset({2})
    assert engines.effective_cgas(caps, _facts(device_cc=cc, s_q=128, has_score_max=False, has_score_sum_exp=False)) >= frozenset({2})
    # The flags are the row's whole claim: a row without them declines the ports before any of the above.
    from dataclasses import replace

    off = replace(caps, score_max=False, score_sum_exp=False)
    assert "score_max output" in engines.mismatch(off, _facts(device_cc=cc))


def test_capabilities_field_is_appended():
    """``score_stats_d_shapes`` was appended after ``thd_pack_gqa_d_shapes``; the Rubin small-tile routing field
    ``dense_only_d_shapes`` (2026-10-08) is appended after it -- the append-only Capabilities record."""
    from cudnn.sdpa.fwd.engines import Capabilities

    names = [f.name for f in fields(Capabilities)]
    # ... the zero-scale and pre-folded-scale fields after it, then the resident-bias / kv_mask fields of the Rubin small tiles
    # and their per-shape TILE_N domain after THOSE.
    assert names[-10:] == [
        "thd_pack_gqa_d_shapes",
        "zero_scale",
        "attn_scale_prefolded_d_shapes",
        "score_stats_d_shapes",
        "dense_only_d_shapes",
        "bias_d_shapes",
        "bias_io_dtype_only",
        "kv_mask",
        "kv_mask_dtypes",
        "tile_ns_by_d_shape",
    ], names[-11:]
    assert Capabilities.__dataclass_fields__["score_stats_d_shapes"].default is None
    assert Capabilities.__dataclass_fields__["dense_only_d_shapes"].default == frozenset()
    assert Capabilities.__dataclass_fields__["bias_d_shapes"].default is None
    assert Capabilities.__dataclass_fields__["kv_mask"].default is False


def test_adapter_signatures_append_the_score_kwargs():
    """Public-API signatures evolve append-only: the constructor's samples and execute's tensors are the last parameters."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl, SdpaFwdDslSm100

    ctor = inspect.signature(SdpaFwdDsl.__init__).parameters
    # ... followed by the resident-block operands of the Rubin small tiles (sample_bias / sample_kv_mask) and their knobs.
    j = list(ctor).index("sample_score_max")
    assert list(ctor)[j:] == [
        "sample_score_max",
        "sample_score_sum_exp",
        "sample_bias",
        "sample_kv_mask",
        "n_chunk",
        "bias_mode",
        "bias_stages",
        "padded_tail_form",
        "dead_subtile_quiet",
        "bias_max_form",
        "softmax_wgs",
    ], list(ctor)[j:]
    assert ctor["sample_score_max"].default is None and ctor["sample_score_sum_exp"].default is None
    ex = inspect.signature(SdpaFwdDslSm100.execute).parameters
    assert list(ex)[-4:] == ["score_max_tensor", "score_sum_exp_tensor", "bias_tensor", "kv_mask_tensor"]
    assert ex["score_max_tensor"].default is None and ex["score_sum_exp_tensor"].default is None
    assert ex["bias_tensor"].default is None and ex["kv_mask_tensor"].default is None


def _desc(shape, dtype, name, stride=None):
    """A BSHD-physical (BHSD-logical) TensorDesc with no storage behind it."""
    from cudnn.api_base import TensorDesc

    b, h, s, d = shape
    stride = tuple(stride) if stride is not None else (s * h * d, d, h * d, 1)
    return TensorDesc(
        dtype=dtype, shape=tuple(shape), stride=stride, stride_order=TensorDesc._compute_stride_order(tuple(shape), stride), device="cuda", name=name
    )


def _fake_cc(monkeypatch, cc):
    """check_support reads the LIVE device; pin it so both arch arms run on any host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cc)


def _api(
    *,
    b=1,
    h=8,
    h_kv=None,
    s=512,
    d=128,
    d_v=None,
    dtype=torch.float16,
    fp8=False,
    score_max=True,
    score_sum_exp=True,
    score_stride=None,
    score_dtype=torch.float32,
    **kw,
):
    """A d128-shaped SdpaFwdDslSm100 built from descriptors only, the score samples declared like Stats."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    h_kv = h if h_kv is None else h_kv
    d_v = d if d_v is None else d_v
    qkv_dtype = torch.float8_e4m3fn if fp8 else dtype
    o_dtype = torch.bfloat16 if fp8 else dtype
    q = _desc((b, h, s, d), qkv_dtype, "q")
    k = _desc((b, h_kv, s, d), qkv_dtype, "k")
    v = _desc((b, h_kv, s, d_v), qkv_dtype, "v")
    o = _desc((b, h, s, d_v), o_dtype, "o")
    # The Stats geometry: compact (b, h, s, 1) unless a declared (padded / permuted) stride is given.
    score_stride = score_stride if score_stride is not None else (h * s, s, 1, 1)
    if score_max:
        kw["sample_score_max"] = _desc((b, h, s, 1), score_dtype, "score_max", stride=score_stride)
    if score_sum_exp:
        kw["sample_score_sum_exp"] = _desc((b, h, s, 1), score_dtype, "score_sum_exp", stride=score_stride)
    if fp8:
        kw.update(pertensor_fp8=True, dtype_o=o_dtype)
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


def _stub_compiled(monkeypatch, api):
    """Let execute() reach its presence contracts without a kernel or a stream."""
    api._compiled_kernel = object()
    monkeypatch.setattr(api, "_get_default_stream", lambda stream: stream)


@pytest.mark.parametrize("cc", [(10, 0), (10, 7)], ids=["sm100", "sm107"])
def test_adapter_check_support_declines_typed(monkeypatch, cc):
    """Rule 8b twins of the rows' claims on the STANDALONE adapter: the accept side first, then each decline typed --
    ValueError for a malformed request, NotImplementedError for "not mine" -- and execute's presence contract both ways."""
    if cc == (10, 7) and _DSL_FLOOR_WHY:
        pytest.skip(_DSL_FLOOR_WHY)  # Rule 7: below the sm_107a DSL floor the Rubin adapter declines typed before any claim
    _fake_cc(monkeypatch, cc)
    api = _api()
    assert api.check_support()
    assert api.score_max_desc is not None and api.score_sum_exp_desc is not None
    assert api._score_strides == {"score_max": None, "score_sum_exp": None}
    assert _api(score_sum_exp=False).check_support() and _api(score_max=False).check_support()
    assert _api(dtype=torch.bfloat16, stats_log2=True).check_support()
    h, s = 8, 512
    strided = _api(score_stride=(h * (s + 7), s + 7, 1, 1))
    assert strided.check_support()
    assert strided._score_strides == {"score_max": (h * (s + 7), s + 7, 1), "score_sum_exp": (h * (s + 7), s + 7, 1)}
    # Malformed requests -> ValueError.
    with pytest.raises(ValueError):
        _api(score_dtype=torch.bfloat16).check_support()
    with pytest.raises(ValueError, match="non-overlapping"):
        _api(b=2, score_stride=(0, s, 1, 1)).check_support()
    # "Not mine" -> NotImplementedError naming the axis.
    for d, d_v in ((192, 128), (256, 256), (512, 512)):
        with pytest.raises(NotImplementedError, match="wired in"):
            _api(d=d, d_v=d_v).check_support()
    if cc == (10, 0):
        with pytest.raises(NotImplementedError, match="wired in"):
            _api(d=64).check_support()  # the native d64 flavor: not claimed
    else:
        # RE-INVERTED 2026-10-08: Rubin routes a d64 graph WITH the outputs past its dense-only 64-wide tile onto the d128
        # envelope, the kernel with the store (the pre-tile route); without them the same graph takes the 64-wide tile.
        api64 = _api(d=64)
        assert api64.check_support() and api64.flavor == (128, 128)
        plain64 = _api(d=64, score_max=False, score_sum_exp=False)
        assert plain64.check_support() and plain64.flavor == (64, 64)
    with pytest.raises(NotImplementedError, match="split"):
        _api(split_kv=2).check_support()
    with pytest.raises((ValueError, NotImplementedError), match="cga"):
        _api(cga=1).check_support()
    with pytest.raises(NotImplementedError, match="f16/bf16"):
        _api(fp8=True).check_support()
    with pytest.raises(NotImplementedError, match="THD"):
        _api(thd=True).check_support()
    # Execute presence contracts, both directions.
    t = torch.empty(1)
    api = _api()
    assert api.check_support()
    _stub_compiled(monkeypatch, api)
    with pytest.raises(ValueError, match="score_max_tensor is required"):
        api.execute(t, t, t, t)
    with pytest.raises(ValueError, match="score_sum_exp_tensor is required"):
        api.execute(t, t, t, t, score_max_tensor=t)
    plain = _api(score_max=False, score_sum_exp=False)
    assert plain.check_support()
    _stub_compiled(monkeypatch, plain)
    with pytest.raises(ValueError, match="sample_score_max"):
        plain.execute(t, t, t, t, score_max_tensor=t)
    with pytest.raises(ValueError, match="sample_score_sum_exp"):
        plain.execute(t, t, t, t, score_sum_exp_tensor=t)


def test_adapter_declines_on_the_other_arch_lines(monkeypatch):
    """The SM120 / SM80 / SM90 adapters never see the samples (their rows decline the ports); the shared constructor still
    records them, and the SM100-family adapter is the only one whose check_support can admit them."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl

    assert "sample_score_max" in inspect.signature(SdpaFwdDsl.__init__).parameters
    _fake_cc(monkeypatch, (12, 0))
    with pytest.raises((ValueError, NotImplementedError)):
        _api().check_support()


# --------------------------------------------------------------------------- tier 2: rendering pins (host trace-compile)

# The probe body: TemplateParams from the arm's kwargs (the compile() keys travel as `__has_*` and are popped), the d128
# module of the arch, compile(...) with only the keys the module accepts -- so the SAME body renders the pre-edit tree
# (whose compile() had no score keys) and this tree.  Keep byte-identical to the renderer that produced the records.
_PROBE_BODY = """
import inspect
from cudnn.frost.buffers import cutedsl_state
_installed, _version = cutedsl_state()
print("DSL", " ".join(_version) if _version else "unknown")
has_lse = params_kw.pop("__has_lse", True)
score_kw = {k[2:]: params_kw.pop(k) for k in ("__has_score_max", "__has_score_sum_exp") if k in params_kw}
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((128, 128), params, rubin=(arch == "sm_107a"))
accepted = inspect.signature(mod.compile).parameters
compile_kw = dict(has_lse=has_lse, lse_kind="dense")
compile_kw.update({k: v for k, v in score_kw.items() if k in accepted})
print("COMPILE_KW", sorted(compile_kw.items()))
mod.compile(**compile_kw)
"""
_COUNTS = {
    "STG": (" STG",),
    "LDTM": ("LDTM",),
    "FSEL": (" FSEL",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "USETMAXREG": ("USETMAXREG",),
}
_RECORDS = {"dense_bf16": {"dtype_qkv": 2}, "causal_f16": {"dtype_qkv": 3, "window_right": 0}, "dense_bf16_nolse": {"dtype_qkv": 2, "__has_lse": False}}
_SCORE_ARM = {"dtype_qkv": 2, "__has_score_max": True, "__has_score_sum_exp": True}
_RENDERINGS = Path(__file__).resolve().parent / "renderings"


def _record(arch):
    """(dsl line, {record: md5}) of the committed pre-edit record ``renderings/md5_fwd_d128_f16_score_stats_off_<arch>.txt``."""
    f = _RENDERINGS / f"md5_fwd_d128_f16_score_stats_off_{arch}.txt"
    assert f.is_file(), f
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(rf"fwd_d128_f16 {arch} (\S+) rc=0 cubin_md5=([0-9a-f]{{32}})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a"])
def test_pre_edit_md5_records_are_committed_and_complete(arch):
    dsl, want = _record(arch)
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert set(want) == set(_RECORDS), (sorted(want), sorted(_RECORDS))


def _skip_unless_the_record_dsl(dsl):
    from cudnn.frost.buffers import cutedsl_state

    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: the cubin is a function of the DSL build, re-render the record")


@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a"])
@pytest.mark.parametrize("record", list(_RECORDS))
def test_no_score_arm_cubin_is_byte_identical_to_the_pre_edit_record(tmp_path, arch, record):
    """With the outputs not requested the d128 kernel traces byte-identically to the tree BEFORE the side outputs landed:
    every new value is a trace-time alias or sits under a const_expr on the None tensors (host trace-compile, no device)."""
    dsl, want = _record(arch)
    _skip_unless_the_record_dsl(dsl)
    probe = run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch=arch, params=_RECORDS[record], tag=f"score_off_{record}")
    assert (
        probe.cubin_md5 == want[record]
    ), f"{arch} {record}: cubin md5 {probe.cubin_md5} != the pre-edit record's {want[record]} -- the no-score rendering changed"


@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a"])
def test_score_arm_adds_exactly_the_stores(tmp_path, arch):
    """The score arm is a different cubin that adds one global store per output per Q sub-tile (2 outputs x TILES_Q = 2)
    and no new spills; the base arm's counts are the pins."""
    base = run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch=arch, params=_RECORDS["dense_bf16"], tag="score_off")
    on = run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch=arch, params=_SCORE_ARM, tag="score_on")
    assert on.cubin_md5 != base.cubin_md5
    assert on.stats["STG"] == base.stats["STG"] + 4, (base.stats, on.stats)
    assert on.stats["LDTM"] == base.stats["LDTM"], "the outputs are aliases of the Stats already loaded: no extra TMEM read"
    assert_no_new_spills(on.stats, base.stats, tag=f"{arch} score arm: ")


# --------------------------------------------------------------------------- tier 3: GPU cells (the row of the device present)

# Tolerances DERIVED from the kernel's arithmetic, not from the first passing run:
#   score_max: S is the fp32 tcgen05 accumulation of exact bf16/f16 products over d = 128 (|err| <= 128 * 2^-24 * sum|q k|,
#              ~2e-4 at these magnitudes), then one FMUL by scale*log2(e) and one by ln 2 (2 roundings, ~1e-7 relative):
#              atol 1e-3 is >= 5x that bound; rtol 1e-5 covers the two roundings.
#   score_sum_exp: a sum of up to S_kv ex2.approx terms (2^-22 relative each) plus the per-KV-tile rescales (one more
#              ex2.approx each) and the fp32 accumulation (S_kv * 2^-24): < 2e-4 relative at S_kv <= 512, < 4e-4 at the
#              n_kv = 8 cell's S_kv = 1024 (1024 * 2^-22 + 1024 * 2^-24 + 8 rescales, every term one-sided); rtol 1e-3.
#   Stats == score_max + ln(score_sum_exp): the kernel evaluates exactly that with lg2.approx (2 ulp): 2e-5 absolute.
_TOL_MAX = dict(atol=1e-3, rtol=1e-5)
_TOL_SUM = dict(atol=0.0, rtol=1e-3)
_TOL_LSE_CONSISTENCY = dict(atol=2e-5, rtol=2e-6)
_TOL_O = dict(atol=5e-2, rtol=3e-2)  # the suite's half-precision O bound


def _bhsd(b, h, s, d, dtype):
    return torch.randn(b, s, h, d, device="cuda", dtype=dtype).transpose(1, 2)


def _ref(q, k, v, *, scale, causal=False, bottom_right=False, seq_q_lens=None, seq_kv_lens=None, sinks=None, window_left=None):
    """fp64 oracle of the whole contract: O, the natural-log LSE, score_max and score_sum_exp per row class."""
    b, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = v.shape
    dev = q.device
    g = h_q // h_kv
    k_ref = k.repeat_interleave(g, dim=1).double()
    v_ref = v.repeat_interleave(g, dim=1).double()
    scores = torch.matmul(q.double(), k_ref.transpose(-1, -2)) * scale
    q_lens = seq_q_lens.flatten().to(device=dev, dtype=torch.int64) if seq_q_lens is not None else torch.full((b,), s_q, dtype=torch.int64, device=dev)
    kv_lens = seq_kv_lens.flatten().to(device=dev, dtype=torch.int64) if seq_kv_lens is not None else torch.full((b,), s_kv, dtype=torch.int64, device=dev)
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    diag = i + (kv_lens - q_lens).view(b, 1, 1, 1) if bottom_right else i
    dead_q = i >= q_lens.view(b, 1, 1, 1)
    masked = dead_q | (j >= kv_lens.view(b, 1, 1, 1))
    if causal:
        masked = masked | (j > diag)
    if window_left is not None:
        # The left window rides the diagonal: keys j < diag - W are masked, W keys below the diagonal stay (the kernel's
        # WINDOW_LEFT = W = the graph's sliding_window_length - 1; the same convention as the sm107 suite's references).
        masked = masked | (j < diag - window_left)
    scores = scores.masked_fill(masked, float("-inf"))
    full = scores if sinks is None else torch.cat([scores, sinks.view(1, h_q, 1, 1).double().expand(b, h_q, s_q, 1)], dim=-1)
    smax = full.max(dim=-1).values  # -inf on a row with no finite column
    ssum = torch.exp(full - smax.unsqueeze(-1)).sum(dim=-1).nan_to_num(0.0)  # exp(-inf - -inf) = NaN on an empty row -> 0
    lse = smax + torch.log(ssum)  # -inf + -inf on an empty row
    o = torch.matmul(torch.softmax(full, dim=-1).nan_to_num(0.0)[..., :s_kv], v_ref)
    dead = dead_q.squeeze(-1)
    return dict(o=o, lse=lse.masked_fill(dead, float("-inf")), smax=smax.masked_fill(dead, float("-inf")), ssum=ssum.masked_fill(dead, 0.0))


def _run(
    q,
    k,
    v,
    *,
    scale,
    sdpa_kwargs=None,
    seq_len_kv=None,
    seq_len_q=None,
    sink=None,
    stats=True,
    outputs=("score_max", "score_sum_exp"),
    layout="contiguous",
    stats_log2=False,
):
    """Build the graph with the requested side outputs, pin the arch's f16 row, launch TWICE (sentinel-filled outputs, the
    two-launch delta) and return the buffers."""
    import cudnn

    b, h_q, s_q, _ = q.shape
    d_v = v.shape[-1]
    dtype = q.dtype
    o_gpu = torch.empty(b, s_q, h_q, d_v, device="cuda", dtype=dtype).transpose(1, 2)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    kw = dict(name="sdpa", q=tq, k=tk, v=tv, generate_stats=stats, attn_scale=scale, stats_use_log2=stats_log2)
    vp = {tq: q, tk: k, tv: v}
    side = {}
    for name in outputs:
        buf = make_dense_stats(b, h_q, s_q, layout)
        t = g.tensor(dim=list(buf.shape), stride=list(buf.stride()), data_type=cudnn.data_type.FLOAT, name=name)
        kw[name] = t
        vp[t] = buf
        side[name] = buf
    if seq_len_kv is not None:
        slk = g.tensor_like(seq_len_kv)
        kw.update(seq_len_kv=slk, use_padding_mask=True)
        vp[slk] = seq_len_kv
        slq_t = seq_len_q if seq_len_q is not None else torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
        slq = g.tensor_like(slq_t)
        kw["seq_len_q"] = slq
        vp[slq] = slq_t
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs or {})
    o, stats_t = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = make_dense_stats(b, h_q, s_q, "contiguous")
        stats_t.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
        vp[stats_t] = stats_gpu
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    # The plan: the row's unsplit, unpacked 2-CTA prefill tile -- the tile the score arm is wired on -- PINNED, never inferred from the
    # ranking: with the ports off the heuristics may rank another tile of the row first (the PackGQA cga1 plan on a 148 / 152-SM part for
    # the GQA shapes), and the bitwise comparisons below must be like for like.  A cga2 plan not in the ranked list is replayed through the
    # public autotune record (the way an autotuner replays it).
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    ours = [i for i, n in enumerate(names) if n == _ROW or n.startswith(_ROW + "[")]
    assert ours, f"no {_ROW} plan in {names}"

    def _prefill_tile(i):
        kn = g.plans[i].knobs
        return (kn.split_kv or 1) == 1 and not kn.pack_gqa and kn.cga in (None, 2)

    idx = next((i for i in ours if _prefill_tile(i)), None)
    if idx is None:
        engine_id, record = g.get_engine_and_knobs_at_index(ours[0])
        record = dict(record)
        record[cudnn.knob_type.TILE_CGA_M] = 2
        record[cudnn.knob_type.PACK_GQA] = 0
        record[cudnn.knob_type.SPLIT_KV] = 1
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
    g.select_plan(idx)
    plan = g.plans[idx]
    assert (plan.knobs.split_kv or 1) == 1 and not plan.knobs.pack_gqa and plan.knobs.cga in (None, 2), plan.knobs
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        for buf in side.values():
            buf.fill_(float("nan"))  # sentinel: an unwritten live row stays NaN
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return {n: buf.clone() for n, buf in side.items()}, o_gpu.clone(), (stats_gpu.clone() if stats_gpu is not None else None)

    first, o1, st1 = launch()
    second, o2, st2 = launch()
    assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    for n in side:
        assert torch.equal(first[n].nan_to_num(-7.0), second[n].nan_to_num(-7.0)), f"two-launch delta on {n}"
    return dict(o=o_gpu, stats=stats_gpu, **side)


def _check(out, ref, *, has_stats=True):
    """The contract on every row: keyless rows EXACTLY -inf / 0 (never NaN), live rows against the fp64 oracle, and the
    LSE identity Stats == score_max + ln(score_sum_exp) evaluated from the kernel's own fp32 outputs."""
    smax = out["score_max"].squeeze(-1)
    ssum = out["score_sum_exp"].squeeze(-1)
    empty = torch.isneginf(ref["smax"])
    assert torch.isneginf(smax[empty]).all(), "an empty row's score_max must be exactly -inf"
    assert (ssum[empty] == 0).all(), "an empty row's score_sum_exp must be exactly 0"
    assert torch.isfinite(smax[~empty]).all() and torch.isfinite(ssum[~empty]).all(), "a live row was left unwritten (sentinel NaN) or non-finite"
    torch.testing.assert_close(smax[~empty].double(), ref["smax"][~empty], **_TOL_MAX)
    torch.testing.assert_close(ssum[~empty].double(), ref["ssum"][~empty], **_TOL_SUM)
    torch.testing.assert_close(out["o"].double(), ref["o"], **_TOL_O)
    if has_stats:
        lse = out["stats"].squeeze(-1)
        assert torch.isneginf(lse[empty]).all()
        torch.testing.assert_close(lse[~empty], (smax + torch.log(ssum))[~empty], **_TOL_LSE_CONSISTENCY)


_CASES = {
    # the degenerate-input matrix: the KV ring sweep n_kv = 1 (the P14 phase-drift shape), 2, 3, 4, 5, 8 -- the sm107 KV ring
    # depth is 4 and the sm100 one 2 (config_sm107.CfgD128.STAGES_KV / config_sm100.CfgD128.STAGES_KV), so each arch sees its
    # depth and depth + 1, and n_kv = 8 lets the carried true max lag the threshold-skipped reference over many tiles -- every
    # one with B*H > 1; tail tiles (S not a multiple of 128: the causal band covers one, the row's synthesized KV padding the
    # dense one, whose Q tail also exercises the OOB-row store guard); GQA; a zero-length KV batch; keyless bottom-right rows;
    # keyless rows past a left window (batch 1: kv_len 200, W = 128 -> rows >= 328 have no key, through the SWA arm of the
    # kernel's keyless predicate)
    "dense_nkv2": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256),
    "dense_nkv1": dict(b=2, h=2, h_kv=2, s_q=128, s_kv=128),
    "dense_nkv3_tail_300": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300),
    "dense_nkv5_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640),
    "dense_nkv8_1024": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024),
    "causal_tail_385": dict(b=1, h=4, h_kv=4, s_q=385, s_kv=385, causal=True),
    "causal_swa128_padded_385": dict(b=2, h=2, h_kv=2, s_q=385, s_kv=385, causal=True, window_left=128, kv_lens=[385, 200]),
    "causal_gqa_8_2": dict(b=2, h=8, h_kv=2, s_q=256, s_kv=256, causal=True),
    "padded_kv_zero_batch": dict(b=3, h=4, h_kv=4, s_q=256, s_kv=256, kv_lens=[256, 0, 100]),
    "bottom_right_keyless": dict(b=3, h=8, h_kv=4, s_q=256, s_kv=256, bottom_right=True, q_lens=[256, 129, 0], kv_lens=[200, 256, 128]),
    "bottom_right_keyless_sink": dict(b=3, h=8, h_kv=4, s_q=256, s_kv=256, bottom_right=True, q_lens=[256, 129, 0], kv_lens=[200, 256, 128], sink=True),
}


def _case_inputs(case, dtype, *, seed=0):
    torch.manual_seed(seed)
    c = _CASES[case]
    d = 128
    q = _bhsd(c["b"], c["h"], c["s_q"], d, dtype)
    k = _bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    v = _bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    kv_lens = torch.tensor(c["kv_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "kv_lens" in c else None
    q_lens = torch.tensor(c["q_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "q_lens" in c else None
    sink = torch.randn(1, c["h"], 1, 1, dtype=torch.float32, device="cuda") if c.get("sink") else None
    causal = bool(c.get("causal") or c.get("bottom_right"))
    sdpa_kwargs = dict(use_causal_mask_bottom_right=True) if c.get("bottom_right") else (dict(use_causal_mask=True) if causal else {})
    window_left = c.get("window_left")
    if window_left is not None:
        sdpa_kwargs["sliding_window_length"] = window_left + 1  # the graph's length L keeps L keys up to the diagonal: W = L - 1
    scale = 1.0 / math.sqrt(d) if not c.get("bottom_right") else 0.75  # 0.75 * log2(e) > 1 overflows a finite mask sentinel (the keyless-row trap)
    ref = _ref(
        q,
        k,
        v,
        scale=scale,
        causal=causal,
        bottom_right=bool(c.get("bottom_right")),
        seq_q_lens=q_lens,
        seq_kv_lens=kv_lens,
        sinks=sink.flatten() if sink is not None else None,
        window_left=window_left,
    )
    return q, k, v, dict(scale=scale, sdpa_kwargs=sdpa_kwargs, seq_len_kv=kv_lens, seq_len_q=q_lens, sink=sink), ref


# every case in bf16; the fp16 member on the dense base and on the ports' documented degenerate contract (the keyless row -inf / 0, the sink
# sink_logit / 1) -- the fp16 O cast is the only place the dtype enters
_DT_IDS = {torch.float16: "fp16", torch.bfloat16: "bf16"}
_ORACLE_CELLS = [(c, torch.bfloat16) for c in _CASES] + [("dense_nkv2", torch.float16), ("bottom_right_keyless_sink", torch.float16)]


@requires_blackwell
@pytest.mark.parametrize("case, dtype", _ORACLE_CELLS, ids=[f"{c}-{_DT_IDS[t]}" for c, t in _ORACLE_CELLS])
def test_score_stats_match_the_fp64_oracle(case, dtype):
    q, k, v, kw, ref = _case_inputs(case, dtype)
    out = _run(q, k, v, **kw)
    _check(out, ref)
    c = _CASES[case]
    if c.get("sink"):
        # Keyless rows with a sink: the sink's mass alone -- max = sink_logit, sum_exp = 1, exactly.
        smax, ssum = out["score_max"].squeeze(-1), out["score_sum_exp"].squeeze(-1)
        sink = kw["sink"].view(c["h"], 1)
        assert torch.equal(smax[0, :, :56], sink.expand(c["h"], 56)) and (ssum[0, :, :56] == 1).all()
        assert torch.isneginf(smax[1, :, 129:]).all() and (ssum[1, :, 129:] == 0).all(), "a trimmed row is dead even with a sink"
    if "window_left" in c:
        # Keyless rows PAST the left window (batch 1: [i - W, i] lies beyond kv_len = 200 from row 328 on) read the SELECT's
        # -inf / 0 through the SWA arm of the kernel's keyless predicate; every row below, and all of batch 0, is live.
        smax, ssum = out["score_max"].squeeze(-1), out["score_sum_exp"].squeeze(-1)
        first_dead = c["kv_lens"][1] + c["window_left"]
        assert torch.isneginf(smax[1, :, first_dead:]).all() and (ssum[1, :, first_dead:] == 0).all(), "rows past the window must be keyless"
        assert torch.isfinite(smax[1, :, :first_dead]).all() and (ssum[1, :, :first_dead] > 0).all(), "a row inside the window is live"
        assert torch.isfinite(smax[0]).all() and (ssum[0] > 0).all(), "the full-length batch has no keyless row"
    if "kv_lens" in c and 0 in c["kv_lens"]:
        empty_b = c["kv_lens"].index(0)
        assert torch.isneginf(out["score_max"][empty_b]).all() and (out["score_sum_exp"][empty_b] == 0).all()
        assert torch.isneginf(out["stats"][empty_b]).all() and (out["o"][empty_b] == 0).all()


@requires_blackwell
def test_score_stats_without_stats_and_each_output_alone():
    """The ports are independent of generate_stats and of each other (each store is its own compile-time arm)."""
    q, k, v, kw, ref = _case_inputs("dense_nkv2", torch.bfloat16)
    both = _run(q, k, v, stats=False, **kw)
    assert both["stats"] is None
    _check(both, ref, has_stats=False)
    only_max = _run(q, k, v, outputs=("score_max",), **kw)
    only_sum = _run(q, k, v, outputs=("score_sum_exp",), **kw)
    assert torch.equal(only_max["score_max"], both["score_max"]) and torch.equal(only_sum["score_sum_exp"], both["score_sum_exp"])
    assert "score_sum_exp" not in only_max and "score_max" not in only_sum


@requires_blackwell
def test_score_stats_stay_natural_under_base2_stats():
    """stats_use_log2 scales the Stats alone: score_max / score_sum_exp keep their natural units, and Stats ==
    log2(e) * (score_max + ln score_sum_exp) from the kernel's own outputs."""
    q, k, v, kw, ref = _case_inputs("causal_gqa_8_2", torch.bfloat16)
    out = _run(q, k, v, stats_log2=True, **kw)
    _check(out, ref, has_stats=False)
    smax, ssum, lse2 = out["score_max"].squeeze(-1), out["score_sum_exp"].squeeze(-1), out["stats"].squeeze(-1)
    torch.testing.assert_close(lse2, (smax + torch.log(ssum)) * math.log2(math.e), **_TOL_LSE_CONSISTENCY)


@requires_blackwell
def test_score_stats_strided_layouts_are_bitwise_the_contiguous_ones():
    """A permuted, gapped (B, H, S, 1) declaration binds through its strides (no hidden copy): bitwise the compact run."""
    q, k, v, kw, ref = _case_inputs("dense_nkv2", torch.bfloat16)
    compact = _run(q, k, v, **kw)
    strided = _run(q, k, v, layout="strided", **kw)
    for name in ("score_max", "score_sum_exp"):
        assert not strided[name].is_contiguous()
        assert torch.equal(strided[name], compact[name]), name
    _check(strided, ref)


@requires_blackwell
def test_score_stats_leave_o_bitwise_and_stats_within_one_rounding():
    """O is bitwise the plan without the outputs (the P / O arithmetic is untouched; both plans on the same 2-CTA tile -- ``_run``
    pins it with the ports off too).  Stats is the same expression, max * ln2 + ln(sum), but the score arm materialises the product
    (it is stored), so ptxas can no longer contract it into the add: one fp32 rounding (a few ulp at |Stats| ~ 8), never more -- the
    pin is 4 ulp of that magnitude."""
    q, k, v, kw, _ = _case_inputs("causal_gqa_8_2", torch.bfloat16)
    with_outputs = _run(q, k, v, **kw)
    without = _run(q, k, v, outputs=(), **kw)
    assert torch.equal(with_outputs["o"], without["o"]), "O must not depend on the side outputs"
    diff = (with_outputs["stats"] - without["stats"]).abs().max().item()
    assert diff <= 4 * 2.0 ** (-23) * 8.0, f"Stats moved by {diff} with the side outputs on: more than an FMA-contraction rounding"


@requires_blackwell
def test_score_stats_graph_declines_an_unwired_flavor_and_lists_no_split_or_cga1_plan():
    """Reject cells on the graph: a d256 graph with the ports is never offered by the row; on a d128 graph with the ports
    every plan the row offers is the unsplit 2-CTA tile (the split / cga=1 candidates are declined by mismatch).  The
    graphs use the backend's own admissible form -- ``score_max`` AND ``score_sum_exp`` without ``stats`` (its composite
    softmax outputs stats alone, the pair alone, or neither) -- so a declined row leaves a servable graph behind."""
    import cudnn

    from cudnn.sdpa.fwd.engines import engine_name

    torch.manual_seed(3)
    for d, want in ((256, False), (128, True)):
        b, h, s = 1, 4, 256
        q, k, v = _bhsd(b, h, s, d, torch.bfloat16), _bhsd(b, h, s, d, torch.bfloat16), _bhsd(b, h, s, d, torch.bfloat16)
        g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
        sm = g.tensor(dim=[b, h, s, 1], stride=[h * s, s, 1, 1], data_type=cudnn.data_type.FLOAT, name="score_max")
        se = g.tensor(dim=[b, h, s, 1], stride=[h * s, s, 1, 1], data_type=cudnn.data_type.FLOAT, name="score_sum_exp")
        o, _ = g.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=False, attn_scale=1.0 / math.sqrt(d), score_max=sm, score_sum_exp=se)
        o.set_output(True).set_dim([b, h, s, d]).set_stride([s * h * d, d, h * d, 1])
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        assert offers_engine(g, engine_name(arch=_ARCH)) is want, (d, want)
        if want:
            names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
            ours = [g.plans[i].knobs for i, n in enumerate(names) if n == _ROW or n.startswith(_ROW + "[")]
            assert ours and all(kn.split_kv in (None, 1) and kn.cga in (None, 2) for kn in ours), ours
