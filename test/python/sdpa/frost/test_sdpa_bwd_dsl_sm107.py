# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm107``: the Rubin (SM 10.7) d=256 bf16 / fp16 SDPA backward chain.

Every capability the row claims gets an ACCEPT test that runs the chain through
the graph API with the engine PINNED against an fp64 oracle (operands
``.double()``: a hand-rolled fp32 reference is a TF32 reference), and every
capability it declines gets a REJECT test that asserts the decline through the
row's own ``mismatch()`` on a REAL graph -- per the engine contract an
unsupported feature is asserted, never skipped, so a row that quietly grows a
capability fails here.  The reject probes fake a cc 10.7 device
(``graph_analyzer._device_cc``), so they run on any box and isolate the
feature decline from the arch decline.

The chain is main kernel (dV in TMEM + a ``[B, H, S_kv, S_q]`` dS workspace)
-> two ``bprop_matmul_blackwell`` GEMMs (dK, dQ) -> a GQA fold, so the accept
tests are end-to-end: the seams (LSE convention, workspace hand-off, K-trim
modes, head chunking) are where a unit test of one launch would not look.

Also here, for BOTH kernel families (this row's f16 body and the fp8 row's
body, ``kernels/sm107/bprop_d256_{f16,fp8}.py``): the host-only static pins
cloned from the forward Rubin suite -- ``DESC_VERSION`` vs the SMEM budget,
every ``SmemTile`` wired to it, ring waits wired to ``SPIN_RING_WAITS``, no
cluster-scope release arrive, no internal ptxas knob -- and the sm_107a
trace-compile SASS pins (register split reached the binary, no spills, no
GPU-scope drain, every TMEM load of a slot precedes the arrive that frees it).
The GPU cases carry ``requires_rubin``; the static pins run everywhere.

Bitwise targets: the pre-port kernel's dV / dS dumps on six shapes under
``frost_dev/results/bwd_d256_sm107/<ref>/*.pt`` (see the README beside them),
discovered by content relative to the checkout (or ``FROST_BWD_D256_REF_DIR``);
absent -> those cases skip with reason "no reference dump".
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

import cudnn
from frost_test_utils import (
    _SM,
    arch_known_to_the_dsl,
    assert_no_new_spills,
    cuda_launch_counts,
    nvdisasm_candidates,
    requires_dsl,
    requires_rubin,
    select_engine,
)

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.fixture(autouse=True)
def _mock_target_for_cross_arch_contracts(monkeypatch):
    # This module probes Rubin rows on non-Rubin hosts too (the analyzer's cc faked to 10.7).  bwd mismatch() now carries the
    # fwd rows' sm_107a DSL gate (AGENTS.md Rule 7), so match the fake device with a fake compiler target -- exactly as the
    # fwd suites do; real Rubin runs use the real build.
    import torch
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


_ENGINE = "sdpa_bwd_sm107"
_FP8_ENGINE = "sdpa_bwd_sm107_fp8"
_FAMILY_NAME = "frost_sdpa_bwd"
_SLOT = 4  # engines/manifest.py: EngineSlot(4, opt_in=True) -> FROST_SDPA_BWD_ID_BASE + 4
_ENGINE_ID = 20_604
_D = 256
_RUBIN_CC = (10, 7)
_SM_RANGE = (107, 119)  # the row mirrors the forward Rubin rows (fwd/engines.py)
_DTYPES = (torch.bfloat16, torch.float16)
_DTYPE_IDS = ("bf16", "fp16")

# ---------------------------------------------------------------------------- tolerances: ONE place (the sm120 suite's pattern)
# Half-storage gradients against the fp64 oracle: the kernel accumulates in fp32 from bf16 / fp16 operands and rounds the
# output once, so the residual is the storage dtype's (fp16 carries 3 more mantissa bits than bf16); rtol covers the
# fp32-accumulation order over S_kv / S_q.  Never widen these to turn a case green -- report the magnitude.
_TOL_COS = 0.9999
_TOL = {torch.bfloat16: dict(atol=5e-2, rtol=5e-2), torch.float16: dict(atol=2e-2, rtol=5e-2)}
# One bf16 rounding step, relative: the GQA fold's fp32 sum of per-Q-head bf16 partials may round once differently
# from the pre-port head-reduce's, so the GQA dV bitwise target is "within one bf16 ulp", the MHA one exact.
_BF16_ULP_REL = 2.0**-7

# ---------------------------------------------------------------------------- the two kernel bodies (plan s1; the fp8 row shares this module's static pins)
# The MXFP8 body (its own module is test_sdpa_bwd_mxfp8_sm107.py) shares
# every family-parametrized static pin and SASS pin below.
_KERNEL_FILES = {"f16": "sm107/bprop_d256_f16.py", "fp8": "sm107/bprop_d256_fp8.py", "mxfp8": "sm107/bprop_d256_mxfp8.py"}
_TEMPLATE_TAGS = {"f16": "sdpa_bwd_sm107_main_f16", "fp8": "sdpa_bwd_sm107_main_fp8", "mxfp8": "sdpa_bwd_sm107_main_mxfp8"}
_FAMILIES = tuple(_KERNEL_FILES)


def _kernel_path(family):
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path

    return _sm100_kernel_path(_KERNEL_FILES[family])


def _kernel_source(family):
    """The kernel's source, or a SKIP naming the loud failure while the port has not landed."""
    path = _kernel_path(family)
    if not os.path.isfile(path):
        pytest.skip(f"kernels/{_KERNEL_FILES[family]} has not landed yet; test_kernel_modules_are_present is the loud failure")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code_lines(src):
    """Source with whole-line comments dropped -- the prose in these modules quotes ``desc_version=1`` when explaining the hazard."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _code_only(src):
    """Source with EVERY string literal and comment blanked (spans replaced by spaces, newlines kept): the pins that look
    for a literal keyword value (``k_dim=1``) must not see the docstrings and error messages that discuss it.  Python 3.12
    (PEP 701) tokenizes an f-string as ``FSTRING_START`` / ``FSTRING_MIDDLE`` / ``FSTRING_END`` instead of one ``STRING``, so
    those kinds are blanked too -- otherwise the fp8 body's ``k_dim=1`` tripwire MESSAGE reaches the ``k_dim=`` pin on a
    3.12 venv while a 3.10 venv passes (c05 vs the A100 box, 2026-09-28).  The replacement FIELDS of an f-string are
    ordinary tokens on 3.12, so the whole span between FSTRING_START and FSTRING_END is blanked (the fork allowlist of
    test_sdpa_bwd_d512_sm107.py saw `{_LAST_DESC_ROOT}` leak out of a _require message on every 3.12 CI lane, #1323)."""
    import io
    import tokenize

    blank = {tokenize.STRING, tokenize.COMMENT} | {
        getattr(tokenize, name) for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END") if hasattr(tokenize, name)
    }
    lines = src.splitlines(keepends=True)
    offs, acc = [], 0
    for ln in lines:
        offs.append(acc)
        acc += len(ln)
    out = list(src)

    def _blank(a, b):
        for i in range(a, b):
            if out[i] != "\n":
                out[i] = " "

    fstart, fend = getattr(tokenize, "FSTRING_START", None), getattr(tokenize, "FSTRING_END", None)
    depth, span_start = 0, None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in blank and tok.type not in (fstart, fend):
            continue  # (ENDMARKER / NEWLINE sit past the last line: no span to compute for them)
        a = offs[tok.start[0] - 1] + tok.start[1]
        b = offs[tok.end[0] - 1] + tok.end[1]
        # PEP 701: the replacement fields of an f-string (`{name}`) are ordinary tokens between FSTRING_START and
        # FSTRING_END, so a 3.12 tokenizer would leave them in the code view that a 3.10 one (one STRING token) blanks.
        # Blank the WHOLE f-string span, nested f-strings included, so both interpreters see the same code lines.
        if fstart is not None and tok.type == fstart:
            if depth == 0:
                span_start = a
            depth += 1
        elif fend is not None and tok.type == fend:
            depth -= 1
            if depth == 0:
                _blank(span_start, b)
                span_start = None
        elif depth == 0 and tok.type in blank:
            _blank(a, b)
    return "".join(out)


def _load_kernel(family, **params):
    """Template-load one body the way the adapter does (``load_template`` + the config module's ``TemplateParams``)."""
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams

    _kernel_source(family)
    params.setdefault("dtype_qkv", DTYPE_BF16 if family == "f16" else DTYPE_E4M3)
    return load_template(_kernel_path(family), TemplateParams(**params), tag=_TEMPLATE_TAGS[family])


def _spec(name=_ENGINE):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == name), None)
    assert spec is not None, f"{name} is not in cudnn.sdpa.bwd.engines.ENGINE_SPECS (plan s7: rows sdpa_bwd_sm107 / sdpa_bwd_sm107_fp8)"
    return spec


def _rubin_only():
    if _SM is None or not (_SM_RANGE[0] <= _SM <= _SM_RANGE[1]):
        pytest.skip("needs a Rubin-line GPU (107 <= SM <= 119), have " + ("none" if _SM is None else f"sm_{_SM}"))


# =========================================================================== registration / capabilities (host)


def test_kernel_modules_are_present():
    """The ONE loud failure while the kernel ports are outstanding: every other kernel-dependent case skips with a reason
    that names this test, so a forgotten kernel cannot pass the suite by absence."""
    missing = [f for f in _FAMILIES if not os.path.isfile(_kernel_path(f))]
    assert not missing, f"kernel templates missing under kernels/: {[_KERNEL_FILES[f] for f in missing]} (plan s1)"


def test_engine_is_registered_and_opt_in():
    from cudnn.engines.engine_ids import FROST_SDPA_BWD_ID_BASE
    from cudnn.engines.manifest import MANIFEST

    _spec()
    fam = next(f for f in MANIFEST if f.name == _FAMILY_NAME)
    assert _ENGINE in fam.slots, f"{_ENGINE} has no manifest slot (never reuse a retired slot; plan s7 says slot {_SLOT})"
    slot = fam.slots[_ENGINE]
    assert slot.slot == _SLOT and slot.opt_in, "new engines stay opt-in until they earn arch coverage + benchmarks"
    assert FROST_SDPA_BWD_ID_BASE + slot.slot == _ENGINE_ID


def test_engine_name_grammar():
    """bwd names carry no phase segment: ``engine_name(arch) -> sdpa_bwd_<arch>``."""
    from cudnn.sdpa.bwd.engines import engine_name

    assert engine_name("sm107") == _ENGINE


def test_capabilities_match_what_is_implemented():
    """The row must not claim anything the adapter would refuse at build, and it must claim what the accept tests run.
    The v1 deferrals (plan Q4 / PR-2c) are pinned False here: each flips together with an ACCEPT test in this module and
    a SUPPORT_MATRIX_TRACKER line -- not alone."""
    c = _spec().capabilities
    assert (c.sm_lo, c.sm_hi) == _SM_RANGE, "the Rubin backward row tiles the SM100 family at the Rubin boundary, like the forward rows"
    assert c.d == frozenset({_D}) and not c.d_envelope and not c.dqk_ge_dv, "exact d_qk == d_v == 256: the bodies hardcode TILE_K = TILE_O = 256"
    assert c.dtypes == frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    assert not c.is_fp8 and not c.is_mxfp8 and c.out_dtypes == frozenset(), "the half row; the fp8 body is its own row"
    assert c.causal and c.bottom_right and c.swa and c.gqa, "the measured feature set of the pre-port kernel (plan Q4)"
    assert not c.right_band_widening, "config_sm107 rejects window_right != 0 (right-band widening is not implemented)"
    # THD / ragged is served on the packed path (test_sdpa_bwd_thd_sm107.py): the blocked workspace is sized from the declared
    # totals at BUILD time, and the backward node has no cu_seq_len port.
    assert c.thd and c.thd_declared_totals and not c.cu_seq_len
    assert not c.bias and not c.dbias
    assert not c.decode, "prefill bodies: a 128-row q tile per iteration; s_q == 1 is out of scope"
    assert c.layouts == frozenset({"bshd"})
    assert not c.tile_ms and not c.tile_ns, "fixed geometry: no tile axis, the heuristics list {} as the complete record"
    for deferred in ("sink", "dsink", "deterministic"):
        assert not getattr(c, deferred), f"{deferred} is deferred to PR-2c (plan Q4): claim it together with its accept test here and the tracker line"
    # Not a deferral the kernel arm could lift alone: a graph padding mask carries seq_len_q AND seq_len_kv (the frontend
    # requires both -- test_padding_mask_graph_always_carries_seq_len_q) and no body threads a per-batch Q length, so the graph
    # form is declined rather than served while ignoring the q lengths.  The half body's per-batch KV lengths are served on
    # the adapter's standalone surface instead (seq_kv_lens_present; the *_per_batch_kv_* tests below).
    assert not c.padded, "padded stays a graph-level decline until a body threads per-batch Q lengths (a padded graph always carries seq_len_q)"
    for never in (
        "dropout",
        "score_mod",
        "paged_kv",
        "alibi",
        "block_mask",
        "rng_dump",
        "score_max",
        "score_sum_exp",
        "dynamic_scale",
        "unfuse_fma",
        "seq_q_trim",
    ):
        assert not getattr(c, never), never


# =========================================================================== graph builders


def _io_dtype(dt):
    return cudnn.data_type.HALF if dt == torch.float16 else cudnn.data_type.BFLOAT16


def _bshd_stride(shape):
    """cuDNN declares logical BHSD; the row wants BSHD-physical storage."""
    b, h, s, d = shape
    return [s * h * d, d, h * d, 1]


def _bhsd_stride(shape):
    b, h, s, d = shape
    return [h * s * d, s * d, d, 1]


def _half_bwd_graph(
    b=2,
    hq=2,
    hkv=None,
    sq=256,
    skv=256,
    d=_D,
    d_v=None,
    dt=torch.bfloat16,
    scale="default",
    *,
    bias=False,
    sink=False,
    padded=False,
    thd=False,
    stride_fn=_bshd_stride,
    **sdpa_kwargs,
):
    """An ``sdpa_backward`` graph over BSHD-physical tensors with every output's dims / strides set (standing in for
    build_operation_graph, so the analyzer can read it host-side).  ``scale=None`` omits attn_scale -- it is optional
    on the graph and the engine must default it to 1/sqrt(d).  Returns ``(graph, {port: tensor}, (dQ, dK, dV))``."""
    hkv = hq if hkv is None else hkv
    d_v = d if d_v is None else d_v
    io = _io_dtype(dt)
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    shq, shk, shv, sho = [b, hq, sq, d], [b, hkv, skv, d], [b, hkv, skv, d_v], [b, hq, sq, d_v]
    t = {n: g.tensor(name=n, dim=sh, stride=stride_fn(sh)) for n, sh in (("q", shq), ("k", shk), ("v", shv), ("o", sho), ("do", sho))}
    if thd:
        # Ragged Stats is packed token-major (stride_h == 1, stride_s == H_q), cuDNN's ragged-Stats recipe.
        t["stats"] = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[sq * hq, 1, hq, 1], data_type=cudnn.data_type.FLOAT)
    else:
        t["stats"] = g.tensor(name="stats", dim=[b, hq, sq, 1], stride=[hq * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    kw = dict(sdpa_kwargs)
    if scale is not None:
        kw["attn_scale"] = 1.0 / math.sqrt(d) if scale == "default" else scale
    if bias:
        t["bias"] = g.tensor(name="bias", dim=[1, 1, sq, skv], stride=[sq * skv, sq * skv, skv, 1])
        kw["bias"] = t["bias"]
    if sink:
        t["sink"] = g.tensor(name="sink", dim=[1, hq, 1, 1], stride=[hq, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
        t["dsink"] = g.tensor(name="dsink", dim=[1, hq, 1, 1], stride=[hq, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
        kw["sink_token"], kw["dSink_token"] = t["sink"], t["dsink"]
    if padded or thd:
        # ``padded="kv"`` declares seq_len_kv ALONE -- what a KV-only padding mask would look like; the frontend refuses it
        # (test_padding_mask_graph_always_carries_seq_len_q), which is why the rows' padded claim is a graph-level decline.
        if padded != "kv" or thd:
            t["seq_len_q"] = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
            kw["seq_len_q"] = t["seq_len_q"]
        t["seq_len_kv"] = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_kv=t["seq_len_kv"])
    if thd:
        for n in ("q", "k", "v", "o", "do", "stats"):
            t[n].set_ragged_offset(g.tensor(name=f"{n}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64))
        kw.update(max_total_seq_len_q=b * sq, max_total_seq_len_kv=b * skv)
    dq, dk, dv = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=t["stats"], **kw)
    for out, sh in ((dq, shq), (dk, shk), (dv, shv)):
        out.set_output(True).set_data_type(io).set_dim(sh).set_stride(stride_fn(sh))
        if thd:
            out.set_ragged_offset(g.tensor(name=f"{out.get_name()}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64))
    if sink:
        t["dsink"].set_output(True)
    return g, t, (dq, dk, dv)


def _build_graph(**kw):
    """The GPU path: validate, build, rank.  The engine is pinned by the caller (``select_engine``)."""
    g, t, outs = _half_bwd_graph(**kw)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    return g, t, outs


# =========================================================================== REJECT -- asserted on REAL graphs, never skipped (host, fake cc 10.7)


def _decline_reason(monkeypatch, engine=_ENGINE, cc=_RUBIN_CC, **kw):
    """Why ``engine`` declines the graph, or None if it would serve it.  Asks the row's own ``mismatch()`` against the
    analyzer's facts of a real graph (the ranked plan list also holds BACKEND plans, which confound "my engine is
    absent"), on a FAKED device so the feature gate -- not the arch gate -- is what answers."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    spec = _spec(engine)
    try:
        g, _t, _outs = _half_bwd_graph(**kw)
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts)


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(dt=torch.float16),
        dict(scale=None),
        dict(use_causal_mask=True),
        dict(use_causal_mask_bottom_right=True),
        dict(use_causal_mask=True, diagonal_band_left_bound=256),
        dict(hq=8, hkv=2),
        dict(hq=16, hkv=1),
        dict(sq=500, skv=500),
        dict(sq=768, skv=1280),
        dict(sq=500, skv=1024, use_causal_mask_bottom_right=True),
    ],
    ids=["dense-bf16", "dense-fp16", "default-scale", "causal", "bottom-right", "swa", "gqa-r4", "mqa-r16", "non-tile-S", "768x1280", "bottom-right-ragged-sq"],
)
def test_served_graph_passes_the_row_probe(monkeypatch, kw):
    """Sanity for the reject tests, and the host-side half of every accept claim: the row's probe admits each graph the
    GPU cases below run."""
    assert _decline_reason(monkeypatch, **kw) is None


@pytest.mark.parametrize("d", [128, 264, 512])
def test_reject_other_head_dims(monkeypatch, d):
    """Exact d=256: the d512 chain owns the band above, the sm100/sm120 flavors the sizes below; 264 is not a tile."""
    assert _decline_reason(monkeypatch, d=d) is not None


def test_reject_rectangular_head_dims(monkeypatch):
    """d_qk != d_v is declined: TILE_K == TILE_O == 256 in the body."""
    assert _decline_reason(monkeypatch, d=384, d_v=320) is not None
    assert _decline_reason(monkeypatch, d=256, d_v=128) is not None


def test_reject_gqa_ratio_not_integer(monkeypatch):
    assert _decline_reason(monkeypatch, hq=6, hkv=4) is not None


def test_reject_bias(monkeypatch):
    assert _decline_reason(monkeypatch, bias=True) is not None


def test_reject_right_band_widening(monkeypatch):
    """``diagonal_band_right_bound`` alone (passing use_causal_mask with it forces the bound back to 0)."""
    assert _decline_reason(monkeypatch, diagonal_band_right_bound=64) is not None


def test_reject_sink(monkeypatch):
    """sink + dSink are deferred (plan Q4): the main kernel is sink-agnostic but the dsink reduction is not wired."""
    assert _decline_reason(monkeypatch, sink=True) is not None


def test_accept_thd(monkeypatch):
    """Packed / ragged Q/K/V IS served on this row (the inverse of the reject this used to be -- test/AGENTS.md: invert, never
    delete): the packed path with the kv-blocked dS workspace, per-sequence lengths from the metadata buffer and per-sequence
    output descriptors.  The numerics and every THD conjunction live in ``test_sdpa_bwd_thd_sm107.py``; the graph builder here
    declares the totals (``max_total_seq_len_*``), which the row requires -- without them it is a typed decline."""
    assert _decline_reason(monkeypatch, thd=True) is None
    assert _decline_reason(monkeypatch, thd=True, use_causal_mask=True) is None
    assert _decline_reason(monkeypatch, thd=True, hq=4, hkv=2) is None


def test_reject_decode_shaped(monkeypatch):
    assert _decline_reason(monkeypatch, sq=1, skv=256) is not None


def test_reject_non_bshd_layout(monkeypatch):
    """A BHSD-contiguous Q is declined by the layout envelope (``layouts == {"bshd"}``): the body derives its
    head / batch strides from BSHD-compact storage.  (Staging of an odd dO is the adapter's business, not the row's.)"""
    assert _decline_reason(monkeypatch, stride_fn=_bhsd_stride) is not None


def test_padding_mask_follows_the_padded_claim(monkeypatch):
    """A REAL graph with a padding mask is declined while the row pins ``padded=False``: the graph carries seq_len_q AND
    seq_len_kv (the frontend requires both) and no body threads a per-batch Q length, so serving it would ignore the q
    lengths.  The half body's per-batch KV lengths are served on the adapter's standalone surface instead
    (``test_adapter_per_batch_kv_lengths*``).  Once a body threads per-batch Q lengths and the claim flips, this test inverts
    (the graph is served) and the poisoned dead-entry graph case below takes over the numerics -- inverted rather than
    deleted so the claim keeps a test on this side of the row."""
    reason = _decline_reason(monkeypatch, padded=True)
    if _spec().capabilities.padded:
        assert reason is None, reason
    else:
        assert reason is not None and "padding" in reason, reason


def test_padding_mask_graph_always_carries_seq_len_q(monkeypatch):
    """The premise behind the graph-level decline: ``use_padding_mask`` with seq_len_kv ALONE is refused by the frontend
    (both lengths are required), and a padded graph's facts carry ``seq_q_t`` -- so "KV-only padding" is not a graph the
    row could be asked for, and ``padded=True`` would commit the row to per-batch Q lengths it does not thread."""
    from cudnn.sdpa import graph_analyzer as ga

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    g, _t, _outs = _half_bwd_graph(padded="kv")
    with pytest.raises(ValueError, match="Padding mask requires"):
        g.validate()
    g, _t, _outs = _half_bwd_graph(padded=True)
    facts = ga.analyze(g)
    assert facts is not None and facts.padded and not facts.thd
    assert facts.seq_q_t is not None and facts.seq_kv_t is not None, "a padded backward graph carries BOTH length tensors"


def test_deterministic_follows_the_claim(monkeypatch):
    """``use_deterministic_algorithm``: INFERRED true (dV in TMEM, dK / dQ as GEMMs over the dS workspace, fixed-order
    fold, no atomics) but claimed only after the two-run bitwise test below has run on the node (plan Q4)."""
    reason = _decline_reason(monkeypatch, use_deterministic_algorithm=True)
    if _spec().capabilities.deterministic:
        assert reason is None, reason
    else:
        assert reason is not None


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (12, 0), (8, 0), (9, 0)], ids=["sm100", "sm103", "sm120", "sm80", "sm90"])
def test_reject_other_arch_lines(monkeypatch, cc):
    """The row serves the Rubin line only; below it the sm100 d256 lowerings (and the native backend) own the shape."""
    reason = _decline_reason(monkeypatch, cc=cc)
    assert reason is not None and f"SM{_SM_RANGE[0]}-{_SM_RANGE[1]}" in reason, reason


def test_reject_quantized_graphs(monkeypatch):
    """The half row must decline every quantized backward (the family gate): per-tensor FP8 E4M3 / E5M2
    ``sdpa_fp8_backward`` and block-scale ``sdpa_mxfp8_backward``."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch
    from test_sdpa_bwd_fp8_sm107 import _build_graph as _fp8_graph
    from test_sdpa_bwd_mxfp8_sm100 import _build_graph as _mxfp8_graph

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    caps = _spec().capabilities
    for fp8 in (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2):
        g = _fp8_graph(fp8=fp8)
        facts = ga.analyze(g)
        assert facts is not None and facts.is_fp8
        assert "serves only" in (mismatch(caps, facts) or ""), str(fp8)
    g, _t, _outs = _mxfp8_graph(1, 2, 2, 256, 256, scale=1.0 / math.sqrt(_D))
    facts = ga.analyze(g)
    assert facts is not None and facts.is_mxfp8
    assert "serves only" in (mismatch(caps, facts) or "")


# =========================================================================== ACCEPT -- fp64 oracle (Rubin)


def _causal_keep(sq, skv, dev="cuda", bottom_right=False, left=None, right=0, causal=True):
    """[S_q, S_kv] keep for the band the graph applies: ``causal`` = the diagonal ``kv <= q + diag (+ right)``, ``left`` =
    the graph's ``diagonal_band_left_bound`` (keeps ``kv >= q + diag - (left - 1)``: ``left`` keys per row, what the analyzer
    hands the rows as ``window_left = left - 1``); ``causal=False`` with ``left`` = a sliding window alone."""
    qi = torch.arange(sq, device=dev).view(-1, 1)
    ki = torch.arange(skv, device=dev).view(1, -1)
    diag = (skv - sq) if bottom_right else 0
    keep = (ki <= qi + diag + right) if causal else torch.ones(sq, skv, dtype=torch.bool, device=dev)
    if left is not None:
        keep &= ki >= qi + diag - (left - 1)
    return keep


def _padded_keep(sq, skv, seq_q_lens, seq_kv_lens, dev="cuda"):
    """[B, 1, S_q, S_kv] keep for per-batch lengths (a length-0 KV entry keeps nothing)."""
    qi = torch.arange(sq, device=dev).view(1, 1, -1, 1)
    ki = torch.arange(skv, device=dev).view(1, 1, 1, -1)
    lq = torch.as_tensor(seq_q_lens, device=dev).view(-1, 1, 1, 1)
    lk = torch.as_tensor(seq_kv_lens, device=dev).view(-1, 1, 1, 1)
    return (qi < lq) & (ki < lk)


def _reference64(q, k, v, do, keep=None, group=1, scale=None):
    """fp64 attention backward on the STORAGE-rounded operands.  ``keep`` is a bool mask broadcastable to
    [B, 1, S_q, S_kv].  GQA groups q heads CONTIGUOUSLY (kv head h serves q heads h*g .. h*g+g-1), the convention the
    analyzer, the adapters and ``sdpa.fp8_ref.gqa_kv_head`` share.  ``scale`` None = 1/sqrt(d); an explicit value is the
    graph's attn_scale (0.0 included: uniform P over the kept keys, dQ = dK = 0, dV = sum(dO) / n_kept)."""
    q64, k64, v64, do64 = (x.double() for x in (q, k, v, do))
    kx = k64.repeat_interleave(group, dim=1) if group > 1 else k64
    vx = v64.repeat_interleave(group, dim=1) if group > 1 else v64
    scale = 1.0 / math.sqrt(q.shape[3]) if scale is None else float(scale)
    s = (q64 @ kx.transpose(-1, -2)) * scale
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse.unsqueeze(-1)).nan_to_num_(0.0)
    o = p @ vx
    delta = (o * do64).sum(-1)
    ds = scale * (do64 @ vx.transpose(-1, -2) - delta.unsqueeze(-1)) * p
    dq = ds @ kx
    dk_q = ds.transpose(-1, -2) @ q64
    dv_q = p.transpose(-1, -2) @ do64
    if group > 1:
        hkv = k.shape[1]
        dk_q = dk_q.view(dk_q.shape[0], hkv, group, dk_q.shape[2], dk_q.shape[3]).sum(2)
        dv_q = dv_q.view(dv_q.shape[0], hkv, group, dv_q.shape[2], dv_q.shape[3]).sum(2)
    all_masked = (~keep.any(-1)).expand(lse.shape) if keep is not None else None
    return o, lse, all_masked, dq, dk_q, dv_q


def _bshd_empty(b, s, h, d, dt, fill=None):
    """A [B, H, S, D] view over BSHD memory -- what the row expects.  ``fill`` poisons it."""
    t = torch.empty(b, s, h, d, device="cuda", dtype=dt)
    if fill is not None:
        t.fill_(fill)
    return t.permute(0, 2, 1, 3)


def _check(name, got, ref, dt):
    assert torch.isfinite(got.float()).all(), f"{name}: non-finite output"
    if not ref.bool().any():
        # An identically-zero reference (attn_scale = 0.0: dS = 0 exactly, so dQ = dK = 0): the cosine of two zero vectors is
        # undefined and torch reads it as 0.0, which the gate below would call a mismatch at max|diff| = 0.  The only honest
        # verdict there is the EXACT zero the kernel owes (a GEMM over an all-zero dS), so that is the gate.
        assert (got.float() == 0).all(), f"{name}: the reference is identically zero, got max|{name}| = {got.float().abs().max().item():.3e}"
        return
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    diff = (got.double() - ref.double()).abs()
    assert cos > _TOL_COS, f"{name}: cos={cos:.6f} (max|diff|={diff.max().item():.3e} at |ref|max={ref.abs().max().item():.3e})"
    torch.testing.assert_close(got.float(), ref.float(), **_TOL[dt], msg=lambda m: f"{name} vs the fp64 oracle: {m}")


class _Run:
    def __init__(self, outs, refs, dt):
        self.outs, self.refs, self.dt = outs, refs, dt

    def check(self):
        for name, got, ref in zip(("dQ", "dK", "dV"), self.outs[0], self.refs):
            _check(name, got, ref, self.dt)
        return self


def _run(
    b=2,
    hq=2,
    hkv=None,
    sq=512,
    skv=512,
    dt=torch.bfloat16,
    keep=None,
    omit_scale=False,
    seed=0,
    poison=None,
    runs=1,
    seq_lens=None,
    ws_poison=None,
    attn_scale=None,
    **sdpa_kwargs,
):
    """Build, PIN the engine, execute ``runs`` times and hand back every run's (dQ, dK, dV) plus the fp64 oracle.
    Inputs are unit normal (the pre-port driver's data), drawn on a CPU generator so the dataset does not depend on the
    GPU's SM count (torch's CUDA Philox lays draws out by grid size).  ``poison`` pre-fills the outputs before every run
    (a zero-initialized output hides a skipped store).  ``ws_poison`` (a BYTE, e.g. ``0xFF`` = NaN in every dtype the chain
    stores) pre-fills the WORKSPACE before every run: a stage that reads a scratch region before writing it -- a stage-3
    GEMM reaching a dS tile the main kernel skipped -- then surfaces as NaN in an output instead of riding a stale zero.
    ``seq_lens=(seq_q_lens, seq_kv_lens)`` adds the padding mask.  ``attn_scale`` (None = 1/sqrt(d) on the graph and the
    oracle alike) declares an EXPLICIT scale on both -- 0.0 included, a valid scale the adapters must preserve.
    ``omit_scale`` leaves attn_scale off the graph (no scaling, 1.0) and pre-scales Q by 1/sqrt(d) so the logits keep
    their usual range."""
    hkv = hq if hkv is None else hkv
    group = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s, h):
        return torch.randn(bb, s, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do = draw(b, sq, hq), draw(b, sq, hq)
    k, v = draw(b, skv, hkv), draw(b, skv, hkv)
    if omit_scale:
        q.mul_(_D**-0.5)
    if seq_lens is not None:
        keep = _padded_keep(sq, skv, *seq_lens) if keep is None else (keep & _padded_keep(sq, skv, *seq_lens))
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group, scale=1.0 if omit_scale else attn_scale)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    if seq_lens is not None:
        sdpa_kwargs["padded"] = True
    graph_scale = None if omit_scale else ("default" if attn_scale is None else attn_scale)
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dt=dt, scale=graph_scale, **sdpa_kwargs)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    dq, dk, dv = _bshd_empty(b, sq, hq, _D, dt), _bshd_empty(b, skv, hkv, _D, dt), _bshd_empty(b, skv, hkv, _D, dt)
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: lse.unsqueeze(-1).contiguous(), dq_t: dq, dk_t: dk, dv_t: dv}
    if seq_lens is not None:
        pack[t["seq_len_q"]] = torch.tensor(seq_lens[0], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
        pack[t["seq_len_kv"]] = torch.tensor(seq_lens[1], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    outs = []
    for _ in range(runs):
        if poison is not None:
            for x in (dq, dk, dv):
                x.fill_(poison)
        if ws_poison is not None:
            ws.fill_(ws_poison)
        g.execute(pack, ws)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    return _Run(outs, (dq_r, dk_r, dv_r), dt)


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_dense(dt):
    _run(dt=dt).check()


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_causal_dtypes(dt):
    """Both dtypes through the masked arm too: the chain reads the dS workspace back in the io dtype, so a dtype mix-up
    shows up here and not in the dense case."""
    _run(dt=dt, keep=_causal_keep(512, 512), use_causal_mask=True).check()


@requires_rubin
def test_causal_bottom_right():
    _run(keep=_causal_keep(512, 512, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_rubin
def test_causal_bottom_right_rectangular():
    """S_kv > S_q shifts the diagonal, which the dK / dQ GEMMs' K-trim has to follow (plan s5: the kv-major workspace
    flips the trim modes relative to the sm100 pairing)."""
    _run(sq=512, skv=1024, keep=_causal_keep(512, 1024, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_rubin
def test_causal_bottom_right_ragged_s_q():
    """Bottom-right with S_q NOT a multiple of the q tile: the diagonal is ``S_kv - S_q`` in REAL rows (1024 - 500 = 524,
    not 1024 - 512).  The f16 body takes ``sq_real`` (``SQ_REAL`` in its problem_size) and the adapter passes
    ``self.s_q_max``, so this row serves it; the fp8 row DECLINES the same shape (its body has no ``seqlen_q_real`` --
    ``test_sdpa_bwd_fp8_sm107.py::test_reject_bottom_right_with_ragged_s_q``).  This case is what keeps the two rows'
    claims honest in opposite directions."""
    _run(b=1, hq=2, sq=500, skv=1024, keep=_causal_keep(500, 1024, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_rubin
def test_sliding_window():
    _run(keep=_causal_keep(512, 512, left=256), use_causal_mask=True, diagonal_band_left_bound=256).check()


@requires_rubin
@pytest.mark.parametrize("hq,hkv", [(4, 2), (8, 1), (16, 1), (8, 4)], ids=["r2", "r8-mqa", "r16-mqa", "r2-h8"])
def test_gqa(hq, hkv):
    """Per-Q-head dK / dV partials folded by ``dkv_reduce`` over whole GQA groups (the head chunk is a multiple of the
    group -- config_sm107.validate_head_chunk)."""
    _run(hq=hq, hkv=hkv, sq=256, skv=256).check()


@requires_rubin
@pytest.mark.parametrize("sq,skv", [(768, 1280), (500, 500), (300, 200), (257, 129), (384, 640)])
def test_non_tile_multiple_seqlens(sq, skv):
    """Neither S_q nor S_kv has to be a multiple of the q tile (128) / the kv block (256): the adapter pads and the
    tail is masked (plan Q9).  (768, 1280) is the pre-port driver's rectangular shape."""
    _run(sq=sq, skv=skv).check()


@requires_rubin
@pytest.mark.parametrize("sq,skv", [(500, 500), (1000, 1000)])
def test_non_tile_multiple_causal(sq, skv):
    _run(sq=sq, skv=skv, keep=_causal_keep(sq, skv), use_causal_mask=True).check()


@requires_rubin
def test_default_attn_scale():
    """attn_scale is OPTIONAL on the graph; omitting it means no scaling (1.0), the backend's meaning."""
    _run(omit_scale=True).check()


@requires_rubin
def test_explicit_zero_attn_scale_is_preserved():
    """An EXPLICIT attn_scale = 0.0 is a valid declared scale, not an omission: P is uniform over the keys, so dQ = dK = 0
    EXACTLY (the kernel's ``dS = attn_scale * P * (dP - delta)`` carries the 0.0) and dV = sum(dO) / S_kv.  The adapter used to
    fold ``== 0.0`` into the None default and run the graph at 1/sqrt(d): max|dQ| 1.14, max|dK| 0.97, dV off by 0.69 at this
    shape (Codex review on #1212).  Stats is the exact LSE of the zero scores, log(S_kv), as the oracle computes it."""
    run = _run(b=1, hq=1, sq=128, skv=256, attn_scale=0.0, poison=float("nan")).check()
    for name, got in zip(("dQ", "dK"), run.outs[0][:2]):
        assert (got.float() == 0).all(), f"{name}: attn_scale = 0.0 must give an EXACT zero, got max |{name}| = {got.float().abs().max().item():.4f}"


@requires_rubin
def test_workspace_is_build_time_honest():
    """get_workspace_size() is a pure function of the shape (dS workspace chunk + delta + GQA partials), not something
    that grows at execute -- what makes the plan CUDA-graph friendly."""
    g, _t, _outs = _build_graph(b=2, hq=2, sq=256, skv=256)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == g.get_workspace_size() > 0


# --------------------------------------------------------------------------- the degenerate matrix, transposed for bprop (sdpa-invariants s9)


@requires_rubin
@pytest.mark.parametrize("n_q_tiles", [1, 2, 3, 4, 5])
def test_q_tiles_per_kv_block(n_q_tiles):
    """q-tiles per kv block = 1, 2, 3, STAGES_Q (2 f16 / 3 fp8), STAGES_Q + 1: the Q / dO rings wrap at STAGES_Q, and
    a ring index conflated with an accumulator parity fails at the first reuse (sdpa-invariants s6)."""
    _run(b=1, hq=1, sq=128 * n_q_tiles, skv=256).check()


@requires_rubin
@pytest.mark.parametrize("n_kv_blocks", [1, 3])
def test_kv_blocks_with_several_tiles_in_flight(n_kv_blocks):
    """kv blocks = 1 and > 1 with B*H > 1: several persistent tiles per CTA, so a phase advance missing on one path drifts
    the next tile (P14)."""
    _run(b=2, hq=2, sq=256, skv=256 * n_kv_blocks).check()


@requires_rubin
def test_causal_tail_kv_block_writes_zeros_not_residue():
    """Top-left causal with S_kv > S_q: no q row attends kv rows >= S_q, so those kv blocks' q-range is EMPTY.  The body
    runs its forced N >= 1 fully-masked tile there (P = 0 -> dS = 0, dV += 0) and must STORE zeros -- a poisoned output
    must come back exactly 0 on those rows, never the poison and never residue * 0 (NaN)."""
    sq, skv = 256, 768
    run = _run(b=2, hq=2, sq=sq, skv=skv, keep=_causal_keep(sq, skv), use_causal_mask=True, poison=float("nan")).check()
    _dq, dk, dv = run.outs[0]
    for name, got in (("dK", dk), ("dV", dv)):
        tail = got[:, :, sq:, :].float()
        assert torch.isfinite(tail).all(), f"{name}: NaN poison survived on the unattended kv rows"
        assert (tail == 0).all(), f"{name}: unattended kv rows must be EXACTLY zero, got max|.|={tail.abs().max().item():.3e}"


# One case per arm of the band the rows serve, on the shapes where the trim's clamps and roundings bite.  ``left`` is the
# graph's band bound (``W = left - 1`` reaches the kernels); ``causal=None`` = a window without a causal diagonal.
_MASK_POISON_CASES = {
    "causal": dict(sq=512, skv=512, causal=True),
    "causal-tl-rect-q": dict(sq=768, skv=512, causal=True),  # q rows past S_kv attend everything; every kv block writes them
    "causal-tl-tail-kv": dict(sq=256, skv=768, causal=True),  # kv blocks past S_q: the forced fully-masked last q tile is what dK's clamp reads
    "bottom-right": dict(sq=512, skv=1024, causal=True, bottom_right=True),  # shift 512: a block multiple
    "bottom-right-ragged-sq": dict(b=1, sq=500, skv=1024, causal=True, bottom_right=True),  # shift 524: the LO floor and the dQ pair straddle
    "bottom-right-ragged-kv": dict(sq=512, skv=1000, causal=True, bottom_right=True),  # shift 488 + the padded-kv arm
    "swa640": dict(sq=1024, skv=1024, causal=True, left=640),  # W = 639: the dK upper / dQ lower bounds at a 64-multiple + 63
    "swa200": dict(sq=1024, skv=1024, causal=True, left=200),  # W = 199: neither a k-tile nor a q-tile multiple
    "swa200-bottom-right": dict(sq=512, skv=1024, causal=True, bottom_right=True, left=200),
    "swa200-bottom-right-ragged-sq": dict(b=1, sq=500, skv=1024, causal=True, bottom_right=True, left=200),
    "swa200-no-causal": dict(sq=1024, skv=1024, causal=None, left=200),  # the diagonal dropped (causal_diag=False)
    "swa200-tl-rect-kv": dict(sq=512, skv=1024, causal=True, left=200),  # top-left window with S_kv > S_q: blocks past S_q run the forced tile
    "swa200-gqa": dict(hq=4, hkv=2, sq=1024, skv=1024, causal=True, left=200),  # the dQ GEMM runs once per group member
}
# A window with S_q > S_kv is refused by the frontend ("Sliding window attention is only supported with max_s_q <= max_s_kv",
# fractal 2026-09-29), so the one geometry `_stage3_needs_zero_fill` still fills for -- a top-left window with
# S_q > roundup(S_kv + W, 256) -- cannot reach the adapter through a graph; it stays as the backstop the host tile walk asserts.


def _mask_case_kw(case):
    """``(run kwargs, keep)`` for one ``_MASK_POISON_CASES`` entry (shared with the fp8 suite's twin through its own builder)."""
    case = dict(case)
    causal, bottom_right, left = case.pop("causal"), case.pop("bottom_right", False), case.pop("left", None)
    kw = dict(use_causal_mask_bottom_right=True) if bottom_right else (dict(use_causal_mask=True) if causal else {})
    if left is not None:
        kw["diagonal_band_left_bound"] = left
    keep = _causal_keep(case["sq"], case["skv"], bottom_right=bottom_right, left=left, causal=bool(causal))
    return case, kw, keep


@requires_rubin
@pytest.mark.parametrize("case", list(_MASK_POISON_CASES), ids=list(_MASK_POISON_CASES))
def test_masked_stage3_reads_only_what_stage2_wrote(case):
    """The stage-3 GEMMs' two-sided K-trim reads ONLY dS tiles the main kernel wrote, under every mask arm the row serves --
    so the per-execute workspace zero-fill is gone (``api_dsl_sm107._stage3_needs_zero_fill``; kept for the untrimmed twin
    and the ``swa128-tl-rect-q-past`` geometry, which this test runs WITH the fill).  The whole workspace is poisoned with
    0xFF (NaN in bf16 / fp16 / fp32) before the execute: a GEMM reaching a tile the kernel skipped -- the LO floor one q
    tile below a bottom-right band, a dQ M tile spanning a q pair only half written, the window's zero tiles -- lands NaN
    in dQ / dK (``_check`` asserts finite first), and the fp64 oracle then holds the values.  The kernel side of the
    contract is the pair rounding in ``_q_loop_bounds`` (``config_sm107.q_write_tiles``).  Every case here runs
    WITHOUT the fill (`_stage3_needs_zero_fill` is False on all of them; the tile walk pins that)."""
    shape, kw, keep = _mask_case_kw(_MASK_POISON_CASES[case])
    _run(keep=keep, poison=float("nan"), ws_poison=0xFF, **shape, **kw).check()


@requires_rubin
def test_dead_padded_entry_is_exactly_zero_when_padded_is_claimed():
    """The GRAPH form of the padded arm's degenerate input: one batch entry with seq_kv_len == 0.  Runs only once the row
    claims ``padded`` (until then ``test_padding_mask_follows_the_padded_claim`` asserts the decline: a padded graph carries
    seq_len_q, which no body threads per batch).  The same case runs TODAY through the adapter's standalone surface --
    ``test_adapter_dead_kv_entry_is_exactly_zero`` (LSE 0 and -inf) -- and the fp8 d256 forward serves it too (the gated
    block's fp8 dead-entry test), so nothing but the graph claim gates this one.  Poisoned outputs: the dead entry's
    dK / dV / dQ must be exactly 0 (a select, not residue * 0), the live entry exact."""
    if not _spec().capabilities.padded:
        pytest.skip("padded is a graph-level decline (a padded graph carries seq_len_q); the adapter-level twin runs")
    b, sq, skv = 2, 256, 512
    run = _run(b=b, hq=2, sq=sq, skv=skv, seq_lens=([sq, sq], [skv, 0]), poison=float("nan")).check()
    for name, got in zip(("dQ", "dK", "dV"), run.outs[0]):
        dead = got[1].float()
        assert torch.isfinite(dead).all() and (dead == 0).all(), f"{name}: the seq_kv_len == 0 entry must be EXACTLY zero"


@requires_rubin
@pytest.mark.parametrize("b", [257, 300])
def test_ragged_kv_batches_past_the_fill_block_read_their_own_kv_length(b):
    """B > 256 on a ragged S_kv: the padded mask arm reads ``seq_kv_lens[b]`` for EVERY batch, and the per-batch kv-length
    fill used to cover one 256-thread block, so batches 256.. read workspace residue.  ``ws_poison=0`` makes the unwritten
    entry read 0 deterministically -- a dead batch whose dQ / dK / dV came back as EXACT zeros (fp64 reference max
    0.78 / 0.93 / 1.13 at B = 257, S_kv = 129; Codex review on #1212).  B = 256, and a tile-multiple S_kv (dense arm, no read)
    at any B, passed all along, so this shape is the pin -- on a poisoned workspace the fill is the only writer of."""
    _run(b=b, hq=1, hkv=1, sq=128, skv=129, ws_poison=0, poison=float("nan")).check()


@requires_rubin
def test_padded_kv_lengths_past_the_fill_block_when_padded_is_claimed():
    """The second reader of per-batch kv lengths, the graph-level padding mask (``seq_len_kv``), at B > 256 with zero-length
    entries among the batches past the fill block.  Gated on the ``padded`` claim exactly like the dead-entry test above (the
    row declines padding masks today, asserted host-side); it activates with the claim, so the B > 256 coverage of that arm
    does not depend on someone remembering it then.  The standalone twin runs today:
    ``test_adapter_per_batch_kv_lengths_past_the_fill_block``."""
    if not _spec().capabilities.padded:
        pytest.skip("padded is a graph-level decline (a padded graph carries seq_len_q); the adapter-level twin runs")
    b, sq, skv = 300, 128, 256
    kv_lens = [(skv, skv // 2, 0)[i % 3] for i in range(b)]
    run = _run(b=b, hq=1, hkv=1, sq=sq, skv=skv, seq_lens=([sq] * b, kv_lens), ws_poison=0, poison=float("nan")).check()
    dead_entries = [i for i in range(b) if kv_lens[i] == 0]
    for name, got in zip(("dQ", "dK", "dV"), run.outs[0]):
        dead = got[dead_entries].float()
        assert torch.isfinite(dead).all() and (dead == 0).all(), f"{name}: every seq_kv_len == 0 entry must be EXACTLY zero"


# --------------------------------------------------------------------------- per-batch kv lengths through the adapter's standalone surface


def _per_batch_keep(sq, skv, kv_lens, causal=False, bottom_right=False, left=None, dev="cuda"):
    """[B, 1, S_q, S_kv] keep for per-batch KV lengths composed with the band the adapter applies: ``ki < len_b``; causal
    ``ki <= qi + diag_b`` with the PER-BATCH bottom-right diagonal ``diag_b = len_b - S_q`` (the kernel's ``_causal_diag`` over
    ``seq_kv_lens[b]``; 0 top-left); a window keeps ``ki >= qi + diag_b - (left - 1)`` (``left`` = the graph's band bound)."""
    qi = torch.arange(sq, device=dev).view(1, 1, -1, 1)
    ki = torch.arange(skv, device=dev).view(1, 1, 1, -1)
    lk = torch.as_tensor(kv_lens, device=dev).view(-1, 1, 1, 1)
    diag = (lk - sq) if bottom_right else torch.zeros_like(lk)
    keep = ki < lk
    if causal:
        keep = keep & (ki <= qi + diag)
    if left is not None:
        keep = keep & (ki >= qi + diag - (left - 1))
    return keep


class _AdapterRun(_Run):
    def __init__(self, outs, refs, dt, api, args, ws, lens):
        super().__init__(outs, refs, dt)
        self.api, self.args, self.ws, self.lens = api, args, ws, lens


def _run_adapter(
    b=3,
    hq=2,
    hkv=None,
    sq=512,
    skv=512,
    dt=torch.bfloat16,
    kv_lens=(512, 300, 0),
    *,
    causal=False,
    bottom_right=False,
    left=None,
    dead_lse=0.0,
    seed=0,
    poison=None,
    ws_poison=None,
    runs=1,
):
    """The standalone surface of ``sdpa_bwd_sm107`` with PER-BATCH kv lengths: construct the adapter over the live tensors
    with ``seq_kv_lens_present=True``, compile, execute with ``seq_kv_lens`` ([B] int32), against the fp64 oracle composing
    the SAME lengths and band (``_per_batch_keep``).  ``dead_lse`` is what the Stats tensor holds on a row with no key
    (``_reference64``'s ``all_masked``): 0.0 (this suite's convention) or ``-inf`` (the forward's contract) -- the kernel
    owes exact zeros either way.  ``left`` is the graph's band bound (the adapter takes ``window_size_left = left - 1``)."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    hkv = hq if hkv is None else hkv
    group = hq // hkv
    kv_lens = list(kv_lens)
    assert len(kv_lens) == b and all(0 <= n <= skv for n in kv_lens), kv_lens
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s, h):
        return torch.randn(bb, s, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do = draw(b, sq, hq), draw(b, sq, hq)
    k, v = draw(b, skv, hkv), draw(b, skv, hkv)
    keep = _per_batch_keep(sq, skv, kv_lens, causal=causal, bottom_right=bottom_right, left=left)
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float().masked_fill(all_masked, float(dead_lse))
    stats = lse.unsqueeze(-1).contiguous()
    dq, dk, dv = _bshd_empty(b, sq, hq, _D, dt), _bshd_empty(b, skv, hkv, _D, dt), _bshd_empty(b, skv, hkv, _D, dt)
    api = SdpaBwdDslSm107(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_do=do,
        sample_stats=stats,
        sample_dq=dq,
        sample_dk=dk,
        sample_dv=dv,
        is_causal=bool(causal),
        causal_bottom_right=bool(bottom_right),
        window_size_left=None if left is None else int(left) - 1,
        scale_softmax=1.0 / math.sqrt(_D),
        seq_kv_lens_present=True,
    )
    api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    args = dict(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, do_tensor=do, stats_tensor=stats, dq_tensor=dq, dk_tensor=dk, dv_tensor=dv)
    outs = []
    for _ in range(runs):
        if poison is not None:
            for x in (dq, dk, dv):
                x.fill_(poison)
        if ws_poison is not None:
            ws.fill_(ws_poison)
        api.execute(**args, workspace=ws, seq_kv_lens=lens)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    return _AdapterRun(outs, (dq_r, dk_r, dv_r), dt, api, args, ws, lens)


def _assert_dead_kv_rows_exactly_zero(run, kv_lens):
    """Every kv row at or past its batch's length: dK / dV EXACTLY zero and finite (a select, never residue * 0); a batch with
    length 0 additionally owes an exactly-zero dQ (every dS row of it is zero)."""
    dq, dk, dv = run.outs[0]
    for bi, n in enumerate(kv_lens):
        for name, got in (("dK", dk), ("dV", dv)):
            tail = got[bi, :, n:, :].float()
            assert torch.isfinite(tail).all(), f"{name}[{bi}]: non-finite past kv length {n}"
            assert (tail == 0).all(), f"{name}[{bi}]: kv rows past the length {n} must be EXACTLY zero, got max|.|={tail.abs().max().item():.3e}"
        if n == 0:
            dead = dq[bi].float()
            assert torch.isfinite(dead).all() and (dead == 0).all(), f"dQ[{bi}]: the seq_kv_len == 0 entry must be EXACTLY zero"


@requires_rubin
@pytest.mark.parametrize("mask", ["dense", "causal"])
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_adapter_per_batch_kv_lengths(dt, mask):
    """The accept matrix of the standalone per-batch kv lengths (dtype x band): lengths [512, 300, 0] on S_kv = 512 -- a
    full entry, one ending at 300 (not a multiple of the 128-row tile), one DEAD (length 0) -- against the fp64 oracle
    composing the same lengths; the rows past each length exactly zero, the dead entry's dQ exactly zero (poisoned outputs)."""
    lens = [512, 300, 0]
    run = _run_adapter(b=3, hq=2, sq=512, skv=512, dt=dt, kv_lens=lens, causal=(mask == "causal"), poison=float("nan")).check()
    _assert_dead_kv_rows_exactly_zero(run, lens)


@requires_rubin
@pytest.mark.parametrize("dead_lse", [0.0, float("-inf")], ids=["lse-0", "lse-neg-inf"])
def test_adapter_dead_kv_entry_is_exactly_zero(dead_lse):
    """The padded arm's degenerate input through the standalone surface: one batch entry with seq_kv_len == 0, its Stats
    rows holding 0 or the forward's ``-inf`` (``exp2(S - (-inf)) = inf`` before the mask: the kernel's P must be a SELECT to
    zero, never ``inf * 0``).  Poisoned outputs: the dead entry's dQ / dK / dV exactly 0, the live entry exact.  This is the
    graph-level ``test_dead_padded_entry_is_exactly_zero_when_padded_is_claimed`` running today."""
    lens = [512, 0]
    run = _run_adapter(b=2, hq=2, sq=256, skv=512, kv_lens=lens, dead_lse=dead_lse, poison=float("nan")).check()
    _assert_dead_kv_rows_exactly_zero(run, lens)


@requires_rubin
def test_adapter_per_batch_kv_lengths_with_a_ragged_s_kv_gqa_and_window():
    """S_kv = 640 is not a 256-multiple (padded to 768): the zero-filled K / V staging and the caller's lengths share the one
    padded-mask arm; GQA folds the per-Q-head partials through the padded staging; a top-left causal window (band bound 200)
    bounds the band from both sides -- the arm every top-left mask takes without the zero-fill."""
    lens = [640, 300, 0]
    run = _run_adapter(b=3, hq=4, hkv=2, sq=384, skv=640, kv_lens=lens, causal=True, left=200, poison=float("nan"), ws_poison=0xFF).check()
    assert run.api._zero_ws is False, "a top-left band does not move with the length: no zero-fill"
    _assert_dead_kv_rows_exactly_zero(run, lens)


@requires_rubin
@pytest.mark.parametrize("chunks", [1, 2], ids=["one-chunk", "two-batch-chunks"])
@pytest.mark.parametrize("left", [None, 200], ids=["no-window", "window-200"])
def test_adapter_per_batch_kv_lengths_bottom_right_fills_the_workspace(monkeypatch, left, chunks):
    """Bottom-right causal with per-batch lengths: the kernel's diagonal is ``len_b - S_q`` per batch (negative for the
    300-length entry: its first 212 query rows have no key and read as dead rows) while the stage-3 K-trim is computed from
    the uniform ``S_kv - S_q``.  Three rules keep the GEMMs reading only zeros or what the kernel wrote, each with a cell here:
    ``_stage3_needs_zero_fill(per_batch_kv=True)`` keeps the dS zero-fill (the trim reaches tiles a shorter batch's band never
    wrote; the workspace is poisoned with 0xFF = NaN so a tile read without the fill lands NaN in dQ / dK);
    ``_stage3_trim_window`` drops the window from the trim (``window-200``: with the uniform window edge the GEMMs SKIPPED the
    shorter batches' live tiles -- dQ cosine 0.65, their max|diff| = max|ref|); and the fill runs ahead of every chunk
    (``two-batch-chunks``: the budget is shrunk so B = 4 runs as two batch chunks x two head chunks, slot 0 holding 1024 then
    700 and slot 1 holding 300 then a DEAD batch -- with one fill per execute the dead batch came back with dQ 2.1 / dK 2.8
    of stale dS and the 700-length one 0.16 / 0.24 off).  Every cell: the oracle composes the same lengths and band, dead rows
    exact zeros."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    b, hq, sq, skv, dt = 4, 2, 512, 1024, torch.bfloat16
    lens = [1024, 300, 700, 0]
    if chunks > 1:
        # `_sm107_chunks` shrinks heads first, then the batch: at this budget (two per-(batch, head) slabs) B = 4, H = 2 chunks
        # as b_chunk = 2, qh_chunk = 1 -- four launches over the one workspace slot pair.
        monkeypatch.setattr(sm107, "_SM107_WS_BUDGET_BYTES", (b // chunks) * sq * skv * torch.tensor([], dtype=dt).element_size())
    run = _run_adapter(b=b, hq=hq, sq=sq, skv=skv, dt=dt, kv_lens=lens, causal=True, bottom_right=True, left=left, poison=float("nan"), ws_poison=0xFF).check()
    assert run.api._zero_ws is True, "bottom-right under per-batch lengths must zero-fill the dS workspace"
    assert (run.api._b_chunk, run.api._qh_chunk) == ((b, hq) if chunks == 1 else (b // chunks, 1)), (run.api._b_chunk, run.api._qh_chunk)
    _assert_dead_kv_rows_exactly_zero(run, lens)


@requires_rubin
def test_adapter_per_batch_kv_lengths_past_the_fill_block():
    """B > 256 with the caller's lengths: the kernel reads ``seq_kv_lens[b]`` for EVERY batch straight from the caller's
    buffer (the one-block fill is not involved), zero-length entries among the batches past 256 included -- the standalone
    twin of ``test_padded_kv_lengths_past_the_fill_block_when_padded_is_claimed``."""
    b, sq, skv = 300, 128, 256
    lens = [(skv, skv // 2, 0)[i % 3] for i in range(b)]
    run = _run_adapter(b=b, hq=1, sq=sq, skv=skv, kv_lens=lens, poison=float("nan")).check()
    _assert_dead_kv_rows_exactly_zero(run, lens)


@requires_rubin
def test_adapter_per_batch_kv_lengths_are_a_plan_fact():
    """The lengths operand is bound by the plan: a plan built with ``seq_kv_lens_present`` refuses an execute without the
    buffer, one built without refuses a buffer, ``seq_q_lens`` is refused on both, and a buffer of the wrong element count
    (B + 1, the prefix-sum form) is refused by the prepared ``bind`` -- every one a ValueError before any stage launches."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    lens = [512, 256]
    run = _run_adapter(b=2, hq=2, sq=256, skv=512, kv_lens=lens, poison=float("nan")).check()
    with pytest.raises(ValueError, match="exactly when"):
        run.api.execute(**run.args, workspace=run.ws)
    with pytest.raises(ValueError, match="seq_q_lens"):
        run.api.execute(**run.args, workspace=run.ws, seq_kv_lens=run.lens, seq_q_lens=run.lens)
    with pytest.raises(ValueError, match="contiguous with"):
        run.api.execute(**run.args, workspace=run.ws, seq_kv_lens=torch.zeros(3, dtype=torch.int32, device="cuda"))
    samples = {"sample_" + name[: -len("_tensor")]: value for name, value in run.args.items()}
    plain = SdpaBwdDslSm107(**samples, scale_softmax=1.0 / math.sqrt(_D))
    plain.check_support()
    plain.compile()
    ws = torch.empty(max(plain.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="exactly when"):
        plain.execute(**run.args, workspace=ws, seq_kv_lens=run.lens)


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_two_launches_are_bitwise_and_race_free(dt):
    """Two probes in one: launch 1 vs 2 is the two-launch race trick (a first-launch / cold-cache race the warm second
    launch masks -- a missing fence_proxy at an SMEM -> async boundary), launch 2 vs 3 the determinism the row must show
    before claiming ``use_deterministic_algorithm`` (dV in TMEM, GEMMs over the workspace, fixed-order fold, no atomics).
    Compared on the raw bits (an int16 view), on a causal GQA shape with several tiles per CTA."""
    run = _run(b=2, hq=4, hkv=2, sq=512, skv=768, dt=dt, keep=_causal_keep(512, 768), use_causal_mask=True, runs=3, poison=float("nan")).check()
    for which, (a, b_) in (("launch 2 vs 1 (race)", (run.outs[1], run.outs[0])), ("launch 3 vs 2 (determinism)", (run.outs[2], run.outs[1]))):
        for name, x, y in zip(("dQ", "dK", "dV"), a, b_):
            n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
            assert n_diff == 0, f"{name} {which}: {n_diff} elements differ, max|diff|={(x.float() - y.float()).abs().max().item():.3e}"


@requires_rubin
def test_unserved_d256_graph_never_surfaces_a_bare_runtime_error():
    """Regression test for the error TYPE: a d256 backward this row declines (deterministic, while deferred) either
    finds another plan or raises ``cudnnGraphNotSupportedError`` -- never the bare RuntimeError a pinned backend config
    that fails to finalize used to fold into (every SDPA harness skips on the typed error and FAILS on anything else).
    Unlike the d512 band, d256 bf16 has a native competitor (backend engine 5, eng5_k14=3_k24=2_k27=0_k38=0_k40=3_k41=2;
    the deterministic ask is served by the backend's own plan),
    so "served" is a legitimate outcome here; the bare RuntimeError is the only forbidden one."""
    try:
        g, _t, _outs = _half_bwd_graph(b=2, hq=2, sq=256, skv=256, use_deterministic_algorithm=True)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()
    except cudnn.cudnnGraphNotSupportedError:
        return


# --------------------------------------------------------------------------- the adapter (api_dsl_sm107): host pins + the stage-3 K-trim bitwise pin


def _adapter_desc(shape, dtype, name):
    """A logical-BHSD TensorDesc over BSHD-physical storage (what lower_dsl_bwd hands the adapter)."""
    from cudnn.api_base import TensorDesc

    b, h, s_, d = shape
    stride = (s_ * h * d, d, h * d, 1)
    return TensorDesc(
        dtype=dtype, shape=shape, stride=stride, stride_order=TensorDesc._compute_stride_order(shape, stride), device=torch.device("cuda", 0), name=name
    )


def _adapter(cls, b=2, hq=2, hkv=None, sq=512, skv=512, dt=torch.bfloat16, grad_dt=None, d=_D, **kw):
    """Construct ``cls`` host-side over BSHD-physical descs (no kernel, no compile).  ``scale_softmax`` defaults to 1/sqrt(d)
    unless ``kw`` carries one (None included -- the graph-omitted case); ``d`` widens the shape for the SM100 d512 adapter."""
    from cudnn.api_base import TensorDesc

    hkv = hq if hkv is None else hkv
    grad_dt = dt if grad_dt is None else grad_dt
    stats = TensorDesc(
        dtype=torch.float32, shape=(b, hq, sq, 1), stride=(hq * sq, sq, 1, 1), stride_order=(3, 2, 1, 0), device=torch.device("cuda", 0), name="stats"
    )
    return cls(
        sample_q=_adapter_desc((b, hq, sq, d), dt, "q"),
        sample_k=_adapter_desc((b, hkv, skv, d), dt, "k"),
        sample_v=_adapter_desc((b, hkv, skv, d), dt, "v"),
        sample_o=_adapter_desc((b, hq, sq, d), dt, "o"),
        sample_do=_adapter_desc((b, hq, sq, d), dt, "dO"),
        sample_stats=stats,
        sample_dq=_adapter_desc((b, hq, sq, d), grad_dt, "dQ"),
        sample_dk=_adapter_desc((b, hkv, skv, d), grad_dt, "dK"),
        sample_dv=_adapter_desc((b, hkv, skv, d), grad_dt, "dV"),
        scale_softmax=kw.pop("scale_softmax", 1.0 / math.sqrt(d)),
        **kw,
    )


@pytest.mark.parametrize("row", ["sm107", "sm107_fp8", "sm100_d512"])
def test_explicit_zero_attn_scale_survives_the_adapters(row):
    """Host pin (no kernel): ``_initialize_implementation`` defaults ``scale_softmax`` ONLY when it is None (attn_scale omitted
    on the graph) and keeps an explicit 0.0 -- the analyzer preserves ``attn_scale=0.0`` as 0.0 and it is a valid declared
    scale (uniform P: dQ = dK = 0, dV = sum(dO) / S_kv).  Both sm107 rows (the fp8 row inherits the initializer) and the SM100
    d512 adapter, whose clause was the identical ``or == 0.0`` one-liner and has no GPU on this host -- the Rubin numerics live
    in ``test_explicit_zero_attn_scale_is_preserved`` (this suite and the fp8 one)."""
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107, SdpaBwdDslSm107Fp8

    e4m3 = torch.float8_e4m3fn
    cls, kw = {
        "sm107": (SdpaBwdDslSm107, dict(dt=torch.bfloat16)),
        "sm107_fp8": (SdpaBwdDslSm107Fp8, dict(dt=e4m3, grad_dt=e4m3)),
        "sm100_d512": (SdpaBwdDslSm100, dict(dt=torch.bfloat16, d=512)),
    }[row]
    d = kw.get("d", _D)
    assert _adapter(cls, scale_softmax=0.0, **kw).scale_softmax == 0.0, "an explicit attn_scale = 0.0 must survive the adapter"
    assert _adapter(cls, scale_softmax=None, **kw).scale_softmax == pytest.approx(1.0 / math.sqrt(d)), "None (omitted) still defaults to 1/sqrt(d)"
    assert _adapter(cls, scale_softmax=0.125, **kw).scale_softmax == 0.125


def test_adapter_chunks_are_divisors_that_fit_the_budget():
    """The dS workspace chunk: a divisor of B and of H_q, a multiple of the GQA group, the LARGEST that fits the budget;
    heads shrink before batches; the fp8 row never chunks the batch (its body has no batch_base); nothing fitting ->
    the smallest legal chunk (an honest oversize), never 0."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import _sm107_chunks

    per = 512 * 512 * 2  # one (batch, head) of a 512 x 512 bf16 workspace
    assert _sm107_chunks(2, 8, 2, 512, 512, 2, budget=16 * per) == (2, 8)
    assert _sm107_chunks(2, 8, 2, 512, 512, 2, budget=8 * per) == (2, 4)
    assert _sm107_chunks(2, 8, 2, 512, 512, 2, budget=3 * per) == (1, 2), "group=2 caps the head chunk at 2, so the batch is chunked next"
    assert _sm107_chunks(2, 8, 2, 512, 512, 2, budget=3 * per, batch_chunking=False) == (2, 2), "fp8: the batch stays whole even when the chunk overflows"
    assert _sm107_chunks(3, 6, 3, 512, 512, 2, budget=per) == (1, 3), "nothing fits: (1, group), still a legal chunk"
    for b, h, g in ((1, 128, 1), (4, 64, 8), (2, 6, 3)):
        bc, hc = _sm107_chunks(b, h, g, 8192, 8192, 2)
        assert b % bc == 0 and h % hc == 0 and hc % g == 0 and bc >= 1 and hc >= 1


def test_stage3_renderings_pair_operand_major_with_trim_mode():
    """Plan s5: the kv-major [S_kv, S_q] workspace flips the operand majors relative to the SM100 chain (dK reads dS
    K-major, dQ reads dS^T M-major) but NOT the causal trim modes (dK trims the low q tiles, dQ the high kv blocks).
    Untrimmed / dense renderings carry CAUSAL_K_NONE on both; the shift and the 256-row kv block ride along."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP32
    from cudnn.sdpa.bwd.api_dsl_sm107 import _stage3_params
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, EPI_DESCALE, EPI_NONE, EPI_QUANT

    dk, dq = _stage3_params(DTYPE_BF16, causal=True, shift=512, gran=256, trim=True, cgrp_tile_mn=(256, 256))
    assert (dk.a_is_m_major, dk.causal_mode) == (False, CAUSAL_K_LO)
    assert (dq.a_is_m_major, dq.causal_mode) == (True, CAUSAL_K_HI)
    assert dk.causal_shift == dq.causal_shift == 512 and dk.causal_gran == dq.causal_gran == 256
    assert dk.b_is_n_major and dq.b_is_n_major and dk.dtype_qkv == DTYPE_BF16
    assert dk.cgrp_tile_mn == dq.cgrp_tile_mn == (256, 256), "the cluster tile rides on BOTH records"
    assert dk.epi_mode == dq.epi_mode == EPI_NONE and dk.dtype_out == dq.dtype_out == -1, "the half row's records carry no epilogue"
    for causal, trim in ((True, False), (False, True), (False, False)):
        dk, dq = _stage3_params(DTYPE_BF16, causal=causal, shift=0, gran=256, trim=trim, cgrp_tile_mn=(256, 256))
        assert dk.causal_mode == dq.causal_mode == CAUSAL_K_NONE, (causal, trim)
        assert (dk.a_is_m_major, dq.a_is_m_major) == (False, True), "the majors are a layout fact, not a mask fact"
    with pytest.raises(TypeError):
        _stage3_params(DTYPE_BF16, causal=False, shift=0, gran=256)  # the cluster tile is REQUIRED: no padded rendering by omission
    # The fp8 row's records (an e4m3 dS workspace): the GQA shape -- dK DESCALE to fp32 true-unit partials (dtype_out = FP32: the
    # fold sums the group in fp32 and rounds once), dQ QUANT into the gradient dtype -- and the MHA shape (both QUANT).
    dk, dq = _stage3_params(DTYPE_E4M3, causal=True, shift=0, gran=256, cgrp_tile_mn=(256, 256), epi_modes=(EPI_DESCALE, EPI_QUANT), dtype_out=DTYPE_E4M3)
    assert (dk.dtype_qkv, dk.epi_mode, dk.dtype_out, dk.causal_mode) == (DTYPE_E4M3, EPI_DESCALE, DTYPE_FP32, CAUSAL_K_LO)
    assert (dq.dtype_qkv, dq.epi_mode, dq.dtype_out, dq.causal_mode) == (DTYPE_E4M3, EPI_QUANT, DTYPE_E4M3, CAUSAL_K_HI)
    dk, dq = _stage3_params(DTYPE_E4M3, causal=False, shift=0, gran=256, cgrp_tile_mn=(256, 256), epi_modes=(EPI_QUANT, EPI_QUANT), dtype_out=DTYPE_BF16)
    assert (dk.epi_mode, dk.dtype_out, dq.epi_mode, dq.dtype_out) == (EPI_QUANT, DTYPE_BF16, EPI_QUANT, DTYPE_BF16)
    # The band's second edge: the graph's window rides on BOTH records (dK's upper q bound, dQ's lower kv bound) with the
    # diagonal kept; a window alone drops the diagonal (causal_diag=False, shift 0) but keeps the modes (they name the axis);
    # no window = the defaults (0 / True) on every record that existed before the fields did; the untrimmed twin is NONE.
    dk, dq = _stage3_params(DTYPE_BF16, causal=True, shift=512, gran=256, cgrp_tile_mn=(256, 256), window=639)
    assert (dk.causal_mode, dk.causal_window, dk.causal_diag, dk.causal_shift) == (CAUSAL_K_LO, 639, True, 512)
    assert (dq.causal_mode, dq.causal_window, dq.causal_diag, dq.causal_shift) == (CAUSAL_K_HI, 639, True, 512)
    dk, dq = _stage3_params(DTYPE_BF16, causal=False, shift=0, gran=256, cgrp_tile_mn=(256, 256), window=199)
    assert (dk.causal_mode, dk.causal_window, dk.causal_diag) == (CAUSAL_K_LO, 199, False), "a window without a causal diagonal still trims"
    assert (dq.causal_mode, dq.causal_window, dq.causal_diag) == (CAUSAL_K_HI, 199, False)
    dk, dq = _stage3_params(DTYPE_BF16, causal=True, shift=0, gran=256, cgrp_tile_mn=(256, 256))
    assert (dk.causal_window, dk.causal_diag, dq.causal_window, dq.causal_diag) == (0, True, 0, True), "no window: the pre-existing rendering"
    dk, dq = _stage3_params(DTYPE_BF16, causal=True, shift=512, gran=256, trim=False, cgrp_tile_mn=(256, 256), window=639)
    assert (dk.causal_mode, dk.causal_window, dk.causal_shift, dq.causal_mode, dq.causal_window) == (
        CAUSAL_K_NONE,
        0,
        0,
        CAUSAL_K_NONE,
        0,
    ), "untrimmed: no band at all"


def test_stage3_band_params_are_validated():
    """The band fields are refused where they would silently render the wrong thing: a window on CAUSAL_K_NONE (the
    full range -- every window-masked zero tile read), a dropped diagonal with no window (a trimmed mode with no bound),
    a shift without the diagonal it offsets, a negative window; and on the THD leg -- served trimmed, per sequence -- a
    CONSTANT shift (the diagonal offset is per sequence there: ``thd_causal_bottom_right``) and that field off the THD leg,
    on an untrimmed mode or without the diagonal it offsets."""
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, MatmulTemplateParams, validate_matmul_params

    ok = dict(causal_gran=256, cgrp_tile_mn=(256, 256))
    thd = dict(thd_varlen=True, thd_rows_kv=True)
    for good in (
        dict(**ok, causal_mode=CAUSAL_K_LO, causal_window=639),
        dict(**ok, causal_mode=CAUSAL_K_HI, causal_window=199, causal_diag=False),
        dict(**ok, causal_mode=CAUSAL_K_LO, causal_window=1, causal_shift=512),
        dict(**ok, causal_mode=CAUSAL_K_LO),
        dict(),
        dict(**ok, **thd, causal_mode=CAUSAL_K_LO),
        dict(**ok, **thd, causal_mode=CAUSAL_K_HI, causal_window=63, thd_causal_bottom_right=True),
        dict(**ok, **thd, causal_mode=CAUSAL_K_HI, causal_window=199, causal_diag=False),
        dict(**ok, **thd, causal_mode=CAUSAL_K_HI, b_head_group=16),
        dict(causal_gran=256, causal_mode=CAUSAL_K_LO, causal_window=5, thd_varlen=True),
        # the SM100 d512 chain's Q-major THD trim (thd_rows_kv False): the diagonal edge, per-sequence bottom-right or not
        dict(causal_gran=256, causal_mode=CAUSAL_K_LO, thd_varlen=True),
        dict(causal_gran=256, causal_mode=CAUSAL_K_HI, thd_varlen=True, thd_causal_bottom_right=True),
    ):
        validate_matmul_params(MatmulTemplateParams(**good))
    for bad, needle in (
        (dict(**ok, causal_mode=CAUSAL_K_NONE, causal_window=5), "needs causal_mode"),
        (dict(**ok, causal_mode=CAUSAL_K_LO, causal_window=-1), "must be >= 0"),
        (dict(**ok, causal_mode=CAUSAL_K_NONE, causal_diag=False), "only means something on a trimmed"),
        (dict(**ok, causal_mode=CAUSAL_K_LO, causal_diag=False), "neither edge"),
        (dict(**ok, causal_mode=CAUSAL_K_HI, causal_diag=False, causal_window=5, causal_shift=3), "must be 0"),
        (dict(**ok, **thd, causal_mode=CAUSAL_K_LO, causal_shift=512), "per sequence"),
        (dict(**ok, causal_mode=CAUSAL_K_LO, thd_causal_bottom_right=True), "requires thd_varlen"),
        (dict(**ok, **thd, causal_mode=CAUSAL_K_NONE, thd_causal_bottom_right=True), "needs a trimmed causal_mode"),
        (dict(**ok, **thd, causal_mode=CAUSAL_K_HI, causal_window=5, causal_diag=False, thd_causal_bottom_right=True), "needs a trimmed causal_mode"),
    ):
        with pytest.raises(ValueError, match=re.escape(needle)):
            validate_matmul_params(MatmulTemplateParams(**bad))


def test_stage3_dq_record_groups_its_b_head_by_the_gqa_group(monkeypatch):
    """Host pin: ``_stage3_params(gqa_group=g)`` puts ``b_head_group = g`` on the dQ record ONLY -- its B = K is the head the
    group's ``g`` Q heads share, so ONE launch covers a whole head chunk; the dK record's B = Q is per Q head and keeps 1 -- and
    1 at MHA, on every record built without the argument (every pre-existing rendering keeps its params) and when
    ``DQ_SINGLE_LAUNCH`` is off, which is read at CALL time (the bitwise pin flips it).  ``validate_matmul_params`` refuses a
    non-positive / non-int value (the THD leg takes the group like the dense one: the packed B descriptor's head extent is
    ``n_head // b_head_group`` and its per-sequence clamp touches only the token extent); the host's ``_dq_launches`` refuses a
    group neither 1 nor the GQA group."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm100 import EPI_DESCALE, EPI_QUANT, MatmulTemplateParams, validate_matmul_params
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

    assert sm107.DQ_SINGLE_LAUNCH, "one dQ launch per chunk is what ships; the pin flips it OFF for the twin"
    assert MatmulTemplateParams().b_head_group == 1, "append-only, defaulted: a record that never spells it renders B per head"
    common = dict(shift=0, gran=256, cgrp_tile_mn=(256, 256))
    for g in (2, 4, 16):
        dk, dq = sm107._stage3_params(DTYPE_BF16, causal=True, gqa_group=g, **common)
        assert (dk.b_head_group, dq.b_head_group) == (1, g), (g, dk, dq)
        assert dq.a_is_m_major and not dk.a_is_m_major, "the majors are untouched"
        validate_matmul_params(dq)
        dk, dq = sm107._stage3_params(DTYPE_BF16, causal=False, gqa_group=g, dq_single_launch=False, **common)
        assert (dk.b_head_group, dq.b_head_group) == (1, 1), "the per-member twin renders B per head"
    dk, dq = sm107._stage3_params(DTYPE_BF16, causal=True, gqa_group=1, **common)
    assert (dk.b_head_group, dq.b_head_group) == (1, 1), "MHA: identical either way"
    dk, dq = sm107._stage3_params(DTYPE_BF16, causal=True, **common)
    assert (dk.b_head_group, dq.b_head_group) == (1, 1), "no gqa_group: the pre-existing records"
    # the fp8 row's records: the dQ QUANT rendering takes the group, the dK DESCALE partials stay per head
    dk, dq = sm107._stage3_params(DTYPE_E4M3, causal=True, epi_modes=(EPI_DESCALE, EPI_QUANT), dtype_out=DTYPE_E4M3, gqa_group=16, **common)
    assert (dk.b_head_group, dk.epi_mode, dq.b_head_group, dq.epi_mode) == (1, EPI_DESCALE, 16, EPI_QUANT)
    validate_matmul_params(dq)
    monkeypatch.setattr(sm107, "DQ_SINGLE_LAUNCH", False)
    dk, dq = sm107._stage3_params(DTYPE_BF16, causal=True, gqa_group=8, **common)
    assert dq.b_head_group == 1, "the module constant is read at CALL time (the pin flips it)"
    dk, dq = sm107._stage3_params(DTYPE_BF16, causal=True, gqa_group=8, dq_single_launch=True, **common)
    assert dq.b_head_group == 8
    with pytest.raises(ValueError, match="gqa_group"):
        sm107._stage3_params(DTYPE_BF16, causal=True, gqa_group=0, **common)
    for bad, needle in (
        (dict(b_head_group=0), "positive int"),
        (dict(b_head_group=-2), "positive int"),
        (dict(b_head_group=True), "positive int"),
        (dict(b_head_group=2.0), "positive int"),
    ):
        with pytest.raises(ValueError, match=re.escape(needle)):
            validate_matmul_params(MatmulTemplateParams(**bad))
    validate_matmul_params(MatmulTemplateParams(b_head_group=2, thd_varlen=True, thd_rows_kv=True, cgrp_tile_mn=(256, 256)))
    assert (_dq_launches(1, 1), _dq_launches(4, 4), _dq_launches(4, 1), _dq_launches(16, 16), _dq_launches(16, 1)) == (1, 1, 4, 1, 16)
    for group, bhg in ((4, 2), (16, 4), (2, 4), (1, 2)):
        with pytest.raises(ValueError, match="b_head_group"):
            _dq_launches(group, bhg)


@pytest.mark.parametrize("hq,hkv,hc,bc", [(8, 2, 8, 2), (32, 2, 32, 1), (32, 2, 16, 1), (12, 4, 12, 1), (4, 4, 4, 2), (16, 1, 16, 1)])
@pytest.mark.parametrize("single", (True, False), ids=("one-launch", "per-member"))
def test_stage3_dq_launches_pair_every_q_head_with_its_k_head(hq, hkv, hc, bc, single):
    """The coordinate arithmetic of the dQ launches on a fake flat batch index, no GPU: the (b, h) fork decodes
    ``l -> (h = l % n_head, b = l // n_head)`` for A (dS) and C (dQ) and hands B (K) ``h // b_head_group`` (``_b_head``) against a B
    descriptor ``n_head // b_head_group`` heads deep; ``prepared_host._stage3`` launches ``group // b_head_group`` times, launch
    ``member`` over the chunk's heads ``member :: n_launch`` against the chunk's ``kv_n`` K heads.  For every (b, h) of every
    launch the K head reached must be ``q_head // group`` -- the GQA convention the oracle, the analyzer and the kernels share --
    at ``b_head_group = group`` (one launch) exactly as at 1 (the per-member loop, the twin), and the rendered module carries the
    constant.  Plain-Python twin of the traced arithmetic; the traced spellings are pinned by source below."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

    group = hq // hkv
    assert hc % group == 0 and hq % hc == 0
    bhg = group if single else 1
    kv_n = hc // group
    n_launch = _dq_launches(group, bhg)
    heads = hc // n_launch  # each launch's n_head (A / C head extent); its B is heads // bhg == kv_n heads deep
    assert heads // bhg == kv_n and heads * n_launch == hc
    assert n_launch == (1 if single else group)
    seen = set()
    for hb in range(0, hq, hc):  # the adapter's head chunks (`head_base`)
        for member in range(n_launch):
            for l in range(heads * bc):  # every flat CLC batch index of the launch's grid
                tile_h, tile_b = l % heads, l // heads  # _decode_bh
                h_b = tile_h // bhg  # _b_head
                assert 0 <= h_b < kv_n, "B's coordinate stays inside its descriptor's head extent"
                q_head = hb + member + tile_h * n_launch  # the dS / dQ head the launch's `_window(..., member, heads, n_launch)` view addresses
                k_head = hb // group + h_b  # k_c = the chunk's K heads from hb // group, B's coordinate inside it
                assert k_head == q_head // group, (hb, member, l, q_head, k_head)
                seen.add((tile_b, q_head))
    assert seen == {(b, h) for b in range(bc) for h in range(hq)}, "every (batch, Q head) is written exactly once across the launches"
    # the rendered dQ module carries the group, and the traced arithmetic is the one modelled above
    mod = _load_stage3(
        a_is_m_major=True,
        causal_mode=CAUSAL_K_HI,
        causal_gran=256,
        causal_shift=0,
        vec_bytes_epi=32,
        dtype_qkv=DTYPE_BF16,
        cgrp_tile_mn=(256, 256),
        b_head_group=bhg,
    )
    assert mod.b_head_group == bhg
    body = _code_only(Path(mod.__file__).read_text())
    assert body.count("tile_h_b = _b_head(tile_h)") == 1, "B's head coordinate goes through _b_head"
    assert body.count("tile_h_a = tile_h") == 1 and "(col, coord_m, tile_h, tile_b)" in body, "A and C keep the decoded head"
    assert "return tile_h if cutlass.const_expr(b_head_group == 1) else tile_h // cutlass.Int32(b_head_group)" in body
    assert "b_h = n_head if cutlass.const_expr(b_head_group == 1) else n_head // b_head_group" in body, "B's descriptor head extent"
    assert body.count("_decode_bh(") == 7, "the flat batch decode stays where it was: the definition + the TMA / MMA / epilogue warps' init and loop-tail sites"


def _k_range_twin(mode, m0, nkt, *, gran, shift, window, diag, tk, cgrp_m):
    """Pure-Python twin of ``bprop_matmul_blackwell._causal_k_range`` -- the same statements over Python ints (every
    dividend is clamped at 0 before its ``//``, as there, so floor and trunc division agree)."""
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_LO, CAUSAL_K_NONE

    assert shift >= 0, "the sm107 rows never see a negative shift (the frontend refuses bottom-right with S_q > S_kv)"
    if mode == CAUSAL_K_NONE:
        return 0, nkt
    if mode == CAUSAL_K_LO:
        k_lo = 0
        if diag:
            k_lo = min(((max(m0 - shift, 0) // gran) * gran) // tk, nkt - 1)
        k_hi = nkt
        if window > 0:
            k_hi = max(min((max(m0 + cgrp_m - shift + window, 0) + tk - 1) // tk, nkt), k_lo + 1)
        return k_lo, k_hi
    k_hi = nkt
    if diag:
        hi = max(((m0 + cgrp_m - 1 + shift) // gran + 1) * gran, gran)
        k_hi = max(min((hi + tk - 1) // tk, nkt), 1)
    k_lo = 0
    if window > 0:
        k_lo = min(max(m0 + shift - window, 0) // tk, k_hi - 1)
    return k_lo, k_hi


def _q_range_twin(k0, *, sq_pad, sq_real, skv_real, window, causal, bottom_right, tile_q=128, kv_blk=256, pair=2):
    """Pure-Python twin of the bodies' ``_q_loop_bounds``: ``tile_dsl.mask.compute_q_loop_bounds`` (causal from below, the
    window from above, the bottom-right anchor on both), the outward rounding to the ``pair`` (``config_sm107.q_write_tiles``)
    and the never-empty clamps -- ``[q_lo, q_hi)`` in q tiles for the kv block at ``k0``.  The f16 body's real-length
    diagonal; the fp8 body's is the padded one, equal wherever it serves bottom-right (S_q % 128 == 0)."""
    n_q = sq_pad // tile_q
    diag = (skv_real - sq_real) if bottom_right else 0
    lo = max((k0 - diag) // tile_q, 0) if causal else 0
    hi = n_q
    if window is not None:
        hi = min(-(-max(k0 + kv_blk + window - diag, 0) // tile_q), n_q)
    if causal or window is not None:
        lo = (lo // pair) * pair
        hi = min(-(-hi // pair) * pair, n_q)
    q_lo = min(lo, n_q - 1)
    return q_lo, max(hi, q_lo + 1)


# (S_q, S_kv, left, causal, bottom_right): the served band arms on the shapes where the roundings and clamps bite (the
# poisoned-workspace cases, the bitwise-pin cases, one-block and ragged shapes, the past-the-window geometry).
_BAND_GEOMETRIES = [
    (512, 512, None, True, False),
    (768, 512, None, True, False),
    (256, 768, None, True, False),
    (500, 500, None, True, False),
    (1000, 1000, None, True, False),
    (129, 256, None, True, False),
    (512, 1024, None, True, True),
    (500, 1024, None, True, True),
    (512, 1000, None, True, True),
    (300, 1000, None, True, True),
    (129, 256, None, True, True),
    (1024, 1024, 640, True, False),
    (1024, 1024, 200, True, False),
    (1024, 1024, 2, True, False),  # W = 1, the narrowest window the rows serve (window_left = left - 1 must be > 0: left = 1 is declined)
    (1024, 1024, 1024, True, False),
    (1000, 1000, 200, True, False),
    (512, 1024, 200, True, True),
    (500, 1024, 200, True, True),
    (512, 1000, 200, True, True),
    (1024, 1024, 200, False, False),
    (1000, 1280, 200, False, False),
    (384, 768, 200, True, True),
    # A window with S_q > S_kv: the frontend refuses these graphs, but the model still has to agree with the adapter's fill rule
    # on them -- (512, 256, W=127) writes every q pair, (1024, 256, W=127) and (2048, 512, W=299) leave pairs no block writes.
    (512, 256, 128, True, False),
    (1024, 256, 128, True, False),
    (1024, 256, 128, False, False),
    (2048, 512, 300, True, False),
]


@pytest.mark.parametrize("tk", (64, 128), ids=("bf16-k64", "fp8-k128"))
def test_stage3_two_sided_band_arithmetic(tk):
    """The K-trim's contract, checked against the mask definition on ~26 geometries x both k tiles with no GPU: for EVERY
    cluster M tile of both GEMMs, (a) the K range COVERS every cell the band keeps (a trim that rounds inward drops
    gradient), (b) every dS tile the range reads was WRITTEN by the kernel's q range (twin of ``_q_loop_bounds``, pair
    rounding included) -- the property the poisoned-workspace tests measure on the device -- and (c) the adapter's
    ``_stage3_needs_zero_fill`` says True exactly where (b) fails (the untrimmed twin aside).  The two twins mirror the
    device code statement by statement; a change to either side that is not mirrored here fails (b) or (c)."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_KV_PAD, _SM107_Q_PAD, _stage3_needs_zero_fill, _stage3_params
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16

    gran, cgrp_m, tile_q = _SM107_KV_PAD, 256, _SM107_Q_PAD
    for sq, skv, left, causal, bottom_right in _BAND_GEOMETRIES:
        window = None if left is None else left - 1  # the analyzer hands the rows window_left = left - 1
        sq_pad, skv_pad = -(-sq // tile_q) * tile_q, -(-skv // gran) * gran
        shift = (skv - sq) if bottom_right else 0
        dk, dq = _stage3_params(DTYPE_BF16, causal, shift, gran, trim=True, cgrp_tile_mn=(cgrp_m, cgrp_m), window=window)
        assert (dk.causal_mode, dq.causal_mode) == (CAUSAL_K_LO, CAUSAL_K_HI)
        common = dict(gran=gran, shift=dk.causal_shift, window=dk.causal_window, diag=dk.causal_diag, tk=tk, cgrp_m=cgrp_m)
        # the band the kernel masks (kept cells) and the tiles it writes (per kv block x q tile)
        qi = torch.arange(sq).view(-1, 1)
        ki = torch.arange(skv).view(1, -1)
        kept = (ki <= qi + shift) if causal else torch.ones(sq, skv, dtype=torch.bool)
        if window is not None:
            kept &= ki >= qi + shift - window
        written = torch.zeros(skv_pad // gran, sq_pad // tile_q, dtype=torch.bool)
        for blk in range(skv_pad // gran):
            lo, hi = _q_range_twin(blk * gran, sq_pad=sq_pad, sq_real=sq, skv_real=skv, window=window, causal=causal, bottom_right=bottom_right)
            written[blk, lo:hi] = True
        written_cells = written.repeat_interleave(tile_q, dim=1)[:, :sq].repeat_interleave(gran, dim=0)[:skv]  # [kv, q]
        all_written = True
        tag = f"S_q={sq} S_kv={skv} left={left} causal={causal} bottom_right={bottom_right} tk={tk}"
        # dK: M = kv (one 256-row block per cluster tile), K = q
        nkt = -(-sq // tk)
        for m0 in range(0, skv, cgrp_m):
            k_lo, k_hi = _k_range_twin(CAUSAL_K_LO, m0, nkt, **common)
            assert 0 <= k_lo < k_hi <= nkt, (tag, m0, k_lo, k_hi)
            rows = slice(m0, min(m0 + cgrp_m, skv))
            kept_q = kept[:, rows].any(dim=1)  # q rows with a kept cell in this kv tile
            outside = torch.ones(sq, dtype=torch.bool)
            outside[k_lo * tk : min(k_hi * tk, sq)] = False
            assert not (
                kept_q & outside
            ).any(), f"dK {tag}: M tile {m0} K range [{k_lo}, {k_hi}) drops kept q rows {torch.nonzero(kept_q & outside).flatten().tolist()[:6]}"
            all_written &= bool(written_cells[rows, k_lo * tk : min(k_hi * tk, sq)].all())
        # dQ: M = q (a 256-row pair per cluster tile), K = kv
        nkt = -(-skv // tk)
        for m0 in range(0, sq, cgrp_m):
            k_lo, k_hi = _k_range_twin(CAUSAL_K_HI, m0, nkt, **common)
            assert 0 <= k_lo < k_hi <= nkt, (tag, m0, k_lo, k_hi)
            rows = slice(m0, min(m0 + cgrp_m, sq))
            kept_kv = kept[rows, :].any(dim=0)
            outside = torch.ones(skv, dtype=torch.bool)
            outside[k_lo * tk : min(k_hi * tk, skv)] = False
            assert not (
                kept_kv & outside
            ).any(), f"dQ {tag}: M tile {m0} K range [{k_lo}, {k_hi}) drops kept kv rows {torch.nonzero(kept_kv & outside).flatten().tolist()[:6]}"
            all_written &= bool(written_cells[k_lo * tk : min(k_hi * tk, skv), rows].all())
        needs = _stage3_needs_zero_fill(causal, window, bottom_right, sq_pad, skv_pad, gran, trim=True, cgrp_tile_m=cgrp_m)
        assert needs == (
            not all_written
        ), f"{tag}: the adapter's zero-fill decision ({needs}) disagrees with the tile walk (every read tile written: {all_written})"
        assert _stage3_needs_zero_fill(
            causal, window, bottom_right, sq_pad, skv_pad, gran, trim=False
        ), f"{tag}: the untrimmed twin reads every tile and needs the fill"
    assert not _stage3_needs_zero_fill(False, None, False, 512, 512, gran), "dense never needs the fill"
    # The wide-tile twin (`STAGE3_D256_TILE = False`: a 512-row cluster M tile over 256-row kv blocks) is not tight on any mask.
    assert _stage3_needs_zero_fill(True, None, False, 512, 512, gran, cgrp_tile_m=512) and _stage3_needs_zero_fill(
        True, 639, False, 8192, 8192, gran, cgrp_tile_m=512
    )
    assert not _stage3_needs_zero_fill(False, None, False, 512, 512, gran, cgrp_tile_m=512), "dense stays fill-free on the wide tile too"
    # W = 0 would be "no window" to the template (causal_window=0) while the kernel's SWA arm keeps one key per row: the rows
    # decline window_left <= 0 (check_support / config_sm107) and the record builder refuses it too, so the two cannot disagree.
    with pytest.raises(ValueError, match="window"):
        _stage3_params(DTYPE_BF16, True, 0, gran, trim=True, cgrp_tile_mn=(cgrp_m, cgrp_m), window=0)
    # the one served-mask family that would need the fill is a top-left window with S_q > roundup(S_kv + W, 256) -- a graph the
    # frontend refuses (max_s_q <= max_s_kv under a window), kept as the backstop
    assert _stage3_needs_zero_fill(True, 127, False, 1024, 256, gran) and not _stage3_needs_zero_fill(True, 127, False, 512, 256, gran)


def test_kernels_round_the_masked_q_range_to_the_stage3_pair():
    """Both bodies write their masked q range in the pair the stage-3 trim is keyed on: ``_Q_WRITE_TILES`` is
    ``config_sm107.q_write_tiles(CFG)`` = ``kv_pad_rows / TILE_N`` (2), ``kv_pad_rows`` is what the adapter hands the GEMMs
    as ``causal_gran`` (``_SM107_KV_PAD``), and ``_q_loop_bounds`` rounds with it (the source pin) -- the three facts
    ``test_stage3_two_sided_band_arithmetic`` assumes.  The dense arm folds the rounding out on both bodies."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_KV_PAD, _SM107_Q_PAD
    from cudnn.sdpa.bwd.config_sm107 import kv_pad_rows, q_pad_rows, q_write_tiles

    for family in _FAMILIES:
        mod = _load_kernel(family, window_right=0, window_left=199, **({"dtype_o": DTYPE_BF16} if family == "fp8" else {}))
        assert mod._Q_WRITE_TILES == q_write_tiles(mod.CFG) == kv_pad_rows(mod.CFG) // mod.CFG.TILE_N == 2, family
        assert kv_pad_rows(mod.CFG) == _SM107_KV_PAD and q_pad_rows(mod.CFG) == _SM107_Q_PAD, "the adapter's trim granularity IS the kernel's kv block"
        assert mod.CFG.SWA_WINDOW == 199
        body = _code_only(_kernel_source(family))
        fn = body[body.index("def _q_loop_bounds(") : body.index("def _mask_p_chunk(")]
        assert "_Q_WRITE_TILES" in fn and "(lo // pair) * pair" in fn.replace("b.lo", "lo") and "+ pair - cutlass.Int32(1)) // pair) * pair" in fn, family
        assert "q_hi = cute.math.max(hi, q_lo + cutlass.Int32(1))" in fn, f"{family}: the never-empty clamp must stay AFTER the rounding"


_STAGE3_MD5_RECORD = Path(__file__).resolve().parent / "renderings" / "md5_stage3_sm100a.txt"


def _renderings_dir():
    """-> the local-only develop PTX md5 list FILE (``frost_dev/results/bwd_d256_sm107/parity/renderings/md5_develop_sm100a.txt``
    of this checkout or of the main checkout -- a worktree's frost_dev is untracked) when it exists, else the committed record."""
    root = Path(__file__).resolve().parents[4]
    roots = [root] + ([root.parents[1]] if root.parent.name == ".worktrees" else [])
    for r in roots:
        f = r / "frost_dev" / "results" / "bwd_d256_sm107" / "parity" / "renderings" / "md5_develop_sm100a.txt"
        if f.is_file():
            return f
    return _STAGE3_MD5_RECORD


def _parse_md5_list(f):
    """-> (dsl line or None, {record: md5}) of one PTX md5 list.  Lines: ``dsl=<distribution> <version>`` and
    ``<tag> sm_100a <record> rc=0 ptx_md5=<md5>``."""
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"\S+ sm_100a (\S+) rc=0 ptx_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def _stage3_md5_record(record):
    """-> (path, dsl line or None, {record: md5}) of the stage-3 PTX md5 list that holds ``record``: the COMMITTED
    ``renderings/md5_stage3_sm100a.txt`` (so the pin gates in every checkout and in CI like the stage-2 record), unless a
    checkout's local-only develop list (``_renderings_dir``) has the record -- that list predates the GQA records, so the
    override is PER RECORD, never a blanket one that would turn the six ``hi_*_gqa<g>`` pins into failures."""
    f_local = _renderings_dir()
    if f_local is not None and f_local != _STAGE3_MD5_RECORD and f_local.is_file():
        dsl, want = _parse_md5_list(f_local)
        if record in want:
            return f_local, dsl, want
    f = _STAGE3_MD5_RECORD
    dsl, want = _parse_md5_list(f) if f.is_file() else (None, {})
    return f, dsl, want


# The SM100 d512 chain's ten stage-3 records EXACTLY as `SdpaBwdDslSm100.compile` spells them (no cgrp_tile_mn, no band
# field): both majors x {dense, causal, causal bottom-right 512, THD} bf16 + the dense fp16 pair.  Names = the recorded list's.
# Plus the GQA dQ records the SM100 chain renders since the single-dQ-launch port (`b_head_group = group`: the suite's
# group 4, the A/B's groups 8 and 16) -- new renderings, pinned from this branch's first rendering.
# The (512, 256)-row twins the tile rule adds are DERIVED below from the rule, never listed by hand.
_SM100_STAGE3_RECORDS = {
    "lo_dense": dict(a_is_m_major=True, causal_mode=0, causal_shift=0, dtype_qkv=2, thd_varlen=False),
    "hi_dense": dict(a_is_m_major=False, causal_mode=0, causal_shift=0, dtype_qkv=2, thd_varlen=False),
    "lo_causal": dict(a_is_m_major=True, causal_mode=1, causal_shift=0, dtype_qkv=2, thd_varlen=False),
    "hi_causal": dict(a_is_m_major=False, causal_mode=2, causal_shift=0, dtype_qkv=2, thd_varlen=False),
    "lo_causal_br512": dict(a_is_m_major=True, causal_mode=1, causal_shift=512, dtype_qkv=2, thd_varlen=False),
    "hi_causal_br512": dict(a_is_m_major=False, causal_mode=2, causal_shift=512, dtype_qkv=2, thd_varlen=False),
    "lo_thd": dict(a_is_m_major=True, causal_mode=0, causal_shift=0, dtype_qkv=2, thd_varlen=True),
    "hi_thd": dict(a_is_m_major=False, causal_mode=0, causal_shift=0, dtype_qkv=2, thd_varlen=True),
    "lo_dense_fp16": dict(a_is_m_major=True, causal_mode=0, causal_shift=0, dtype_qkv=3, thd_varlen=False),
    "hi_dense_fp16": dict(a_is_m_major=False, causal_mode=0, causal_shift=0, dtype_qkv=3, thd_varlen=False),
}
for _g in (4, 8, 16):
    _SM100_STAGE3_RECORDS[f"hi_dense_gqa{_g}"] = dict(a_is_m_major=False, causal_mode=0, causal_shift=0, dtype_qkv=2, thd_varlen=False, b_head_group=_g)
    _SM100_STAGE3_RECORDS[f"hi_causal_gqa{_g}"] = dict(a_is_m_major=False, causal_mode=2, causal_shift=0, dtype_qkv=2, thd_varlen=False, b_head_group=_g)
# The THD causal K-trim records (per-sequence trim, `api_dsl.THD_STAGE3_TRIM`): top-left (the constant shift) and
# bottom-right (`thd_causal_bottom_right`: the kernel reads `S_kv[b] - S_q[b]` per sequence) -- renderings the SM100
# chain never produced before `THD_STAGE3_TRIM`, pinned from their first (twice-identical) rendering.
_SM100_STAGE3_RECORDS["lo_thd_causal"] = dict(a_is_m_major=True, causal_mode=1, causal_shift=0, dtype_qkv=2, thd_varlen=True)
_SM100_STAGE3_RECORDS["hi_thd_causal"] = dict(a_is_m_major=False, causal_mode=2, causal_shift=0, dtype_qkv=2, thd_varlen=True)
_SM100_STAGE3_RECORDS["lo_thd_causal_br"] = dict(a_is_m_major=True, causal_mode=1, causal_shift=0, dtype_qkv=2, thd_varlen=True, thd_causal_bottom_right=True)
_SM100_STAGE3_RECORDS["hi_thd_causal_br"] = dict(a_is_m_major=False, causal_mode=2, causal_shift=0, dtype_qkv=2, thd_varlen=True, thd_causal_bottom_right=True)

_SM100_STAGE3_BASE_RECORDS = dict(_SM100_STAGE3_RECORDS)


def _stage3_default_tiles(thd: bool) -> list:
    """Every ``cgrp_tile_mn`` the SM100 chain's rule (`api_dsl._sm100_stage3_cgrp_tile_mn`) can hand a record with this THD
    flag, over the sequence lengths and compute capabilities it keys on -- the rule, not a hand-kept list, decides which
    renderings are DEFAULT renderings and therefore must be pinned."""
    from cudnn.sdpa.bwd import api_dsl

    bound = api_dsl._SM100_STAGE3_SMALL_S_MAX
    lo, hi = api_dsl._SM100_STAGE3_SMALL_S_CC
    ccs = [divmod(x, 10) for x in range(lo, hi + 1)] + [(10, 7), (11, 0), (12, 0), (9, 0)]
    return sorted({api_dsl._sm100_stage3_cgrp_tile_mn(s, thd, cc) for s in (128, bound, bound + 128, 1 << 20) for cc in ccs})


# Plus every OTHER row the rule can choose for each of those records (today: the (512, 256) row for the eight BSHD records,
# what the chain renders at padded max(S_q, S_kv) <= 4096 on cc 10.0 .. 10.6); named ``<record>_<m>x<n>``.  JSON turns the
# tuple into a list, the probe tuple-izes it back.
for _name, _rec in list(_SM100_STAGE3_BASE_RECORDS.items()):
    for _tile in _stage3_default_tiles(_rec["thd_varlen"]):
        if _tile != (512, 512):
            _SM100_STAGE3_RECORDS[f"{_name}_{_tile[0]}x{_tile[1]}"] = dict(_rec, cgrp_tile_mn=_tile)
_SM100_PTX_PROBE = textwrap.dedent(r"""
    import glob, hashlib, json, os, sys
    dump, params_json = sys.argv[1], sys.argv[2]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx"
    os.environ["CUTE_DSL_ARCH"] = "sm_100a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    import cutlass.cute as cute
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_FP16
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm100 import MatmulTemplateParams
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(params_json).items()}  # JSON lists -> the record's tuples
    params = MatmulTemplateParams(b_is_n_major=True, causal_gran=256, vec_bytes_epi=32, **kw)
    mod = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), params, tag="ptx_probe_sm100_stage3")
    print("CONST causal_window", mod.causal_window, "causal_diag", mod.causal_diag, "b_head_group", mod.b_head_group)
    io = cutlass.Float16 if int(params.dtype_qkv) == DTYPE_FP16 else cutlass.BFloat16
    # The recorded list's probe shapes (gemm_probe.py): the THD renderings fold the probe's strides into their setup kernel, so
    # a different D or S changes THEIR md5 (the eight dense / causal ones do not depend on it).
    S_Q, S_KV, H, B, D = 1024, 1024, 8, 1, 256
    a_m_major = bool(params.a_is_m_major)

    @cute.jit
    def probe(entry: cutlass.Constexpr, a_ptr: cute.Pointer, b_ptr: cute.Pointer, c_ptr: cute.Pointer, meta_ptr: cute.Pointer, desc_ptr: cute.Pointer, stream):
        if cutlass.const_expr(a_m_major):
            a = cute.make_tensor(a_ptr, cute.make_layout((S_KV, S_Q, H, B), stride=(1, S_KV, S_KV * S_Q, H * S_KV * S_Q)))
            b = cute.make_tensor(b_ptr, cute.make_layout((D, S_Q, H, B), stride=(1, H * D, D, S_Q * H * D)))
            c = cute.make_tensor(c_ptr, cute.make_layout((S_KV, D, H, B), stride=(H * D, 1, D, S_KV * H * D)))
        else:
            a = cute.make_tensor(a_ptr, cute.make_layout((S_Q, S_KV, H, B), stride=(S_KV, 1, S_KV * S_Q, H * S_KV * S_Q)))
            b = cute.make_tensor(b_ptr, cute.make_layout((D, S_KV, H, B), stride=(1, H * D, D, S_KV * H * D)))
            c = cute.make_tensor(c_ptr, cute.make_layout((S_Q, D, H, B), stride=(H * D, 1, D, S_Q * H * D)))
        meta = cute.make_tensor(meta_ptr, cute.make_layout((1,), stride=(1,)))
        desc = cute.make_tensor(desc_ptr, cute.make_layout((1,), stride=(1,)))
        problem = tuple(cutlass.Int64(x) for x in (a.shape[0], b.shape[0], a.shape[1], H, B, *a.stride, *b.stride, *c.stride, b.shape[1], a.shape[0], c.shape[0]))
        entry(problem, a, b, c, meta, desc, stream)

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    cute.compile(probe, mod._host, ptr(io), ptr(io), ptr(io), ptr(cutlass.Int32, 4), ptr(cutlass.Int64, 8),
                 cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi --gpu-arch sm_100a")
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    if not ptxs:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    print("PTX_MD5", hashlib.md5(open(ptxs[-1], "rb").read()).hexdigest())
""")


def test_stage3_tile_rule_keeps_the_wide_row_off_the_sm100_line():
    """``api_dsl._sm100_stage3_cgrp_tile_mn`` has a compute-capability term: the (512, 256) stage-3 row it hands the SM100 d512
    chain at padded S <= 4096 was measured on the B200 only (148 SMs, 34 vs 74 resident clusters at 231 KiB/CTA), so on cc 10.7
    (the d512 row there inherits ``SdpaBwdDslSm100.compile``) and on cc 11.0 the rule returns the (512, 512) row at S 2048
    dense -- the contrast on cc 10.0 is (512, 256).  Faked-cc host pin, the pattern of the reject probes above; it sits beside
    the md5 pin because this module is not arch-gated and runs on every lane."""
    from cudnn.sdpa.bwd import api_dsl

    for cc in ((10, 7), (11, 0), (12, 0), (9, 0)):
        assert api_dsl._sm100_stage3_cgrp_tile_mn(2048, False, cc) == (512, 512), cc
        assert api_dsl._sm100_stage3_cgrp_tile_mn(128, False, cc) == (512, 512), cc
    assert api_dsl._sm100_stage3_cgrp_tile_mn(2048, False, (10, 0)) == (512, 256)
    assert api_dsl._sm100_stage3_cgrp_tile_mn(2048, False, (10, 6)) == (512, 256)


def test_stage3_md5_record_is_committed_and_complete():
    """The pin's baseline is in the tree (the previous record was local-only, so the pin skipped on every lane): a DSL
    line and exactly the record names the probe renders -- and that set is DERIVED from the chain's tile rule
    (`_stage3_default_tiles`), not hand-kept: the TWENTY base records `compile` spells at the (512, 512) row (the ten MHA
    ones, the six GQA dQ ones with ``b_head_group`` 4 / 8 / 16, the four THD causal-trim ones) plus one ``<record>_<m>x<n>``
    per other row the rule can return for that record (the (512, 256) row for the fourteen BSHD records; the THD records stay
    on the wide row).  A rule that grows a row, or a record that loses its (512, 256) twin in the file, fails here; the lane
    that landed the rule pinned 2 of its 8 new default renderings and the hand-kept set did not notice."""
    from cudnn.sdpa.bwd import api_dsl

    assert _STAGE3_MD5_RECORD.is_file(), _STAGE3_MD5_RECORD
    dsl, want = _parse_md5_list(_STAGE3_MD5_RECORD)
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert len(_SM100_STAGE3_BASE_RECORDS) == 20 and not any("cgrp_tile_mn" in r for r in _SM100_STAGE3_BASE_RECORDS.values())
    assert _stage3_default_tiles(False) == [(512, 256), (512, 512)] and _stage3_default_tiles(True) == [(512, 512)]
    assert api_dsl._SM100_STAGE3_SMALL_S_TILE in _stage3_default_tiles(False)
    expected = set(_SM100_STAGE3_BASE_RECORDS)
    for name, rec in _SM100_STAGE3_BASE_RECORDS.items():
        for tile in _stage3_default_tiles(rec["thd_varlen"]):
            if tile != (512, 512):
                expected.add(f"{name}_{tile[0]}x{tile[1]}")
    assert set(_SM100_STAGE3_RECORDS) == expected, (sorted(_SM100_STAGE3_RECORDS), sorted(expected))
    assert len(_SM100_STAGE3_RECORDS) == 34  # 20 base + 14 (512, 256) twins (the 4 THD records have none)
    assert set(want) == expected, (sorted(want), sorted(expected))


@pytest.mark.parametrize("record", list(_SM100_STAGE3_RECORDS))
def test_stage3_sm100_renderings_ptx_md5_match_the_recorded_develop_list(tmp_path, record):
    """The SM100 d512 chain's stage-3 renderings are PTX-IDENTICAL to the recorded ones -- the ten (512, 512)-row renderings
    to the pre-edit tree's (d4b024671), the (512, 256)-row twins the tile rule derives to their first rendering, the six GQA dQ
    records (``b_head_group`` 4 / 8 / 16: the arm the SM100 chain renders for its dQ GEMM under GQA since ``api_dsl.DQ_SINGLE_LAUNCH``)
    and the four THD causal-trim records to their first (twice-identical) rendering: every field appended to
    ``MatmulTemplateParams`` (``cgrp_tile_mn``, the fp8 arm, ``causal_window`` / ``causal_diag``, ``b_head_group``,
    ``block_scale``, ``thd_rows_kv``, ``thd_causal_bottom_right``) and every ``_TileRow`` edit defaults to what the SM100
    adapter never spelled before, so the
    pre-existing records render byte-for-byte what they always did.  Compared against the COMMITTED record
    (``renderings/md5_stage3_sm100a.txt``; a local ``frost_dev/.../md5_develop_sm100a.txt`` overrides it PER RECORD for
    re-rendering experiments); skips only when the installed DSL build is not the one the record names (the PTX text is a
    function of it).  The rendering is a host trace-compile for sm_100a of the exact record (the test harness itself still
    needs a CUDA device -- run the pin in a GPU slot like any other test).  A PTX md5, not a cubin one: ptxas renames uniform
    registers run to run.  RED-proven: ``ab_stages`` 4 -> 3 on the (512, 512) row fails ``lo_dense`` (1c0477dc... != the recorded
    34bc8347...) and ``lo_dense_fp16`` (9844cb44... != 55a9ed26...) and nothing else -- only the (512, 512) renderings flip."""
    import json

    from cudnn.frost.buffers import cutedsl_state

    f, dsl, want = _stage3_md5_record(record)
    assert f.is_file(), f"no stage-3 PTX md5 list ({f})"
    assert record in want, f"{record} is not in the recorded list ({sorted(want)}) of {f}"
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")
    if not arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    dump = tmp_path / f"sm100a_stage3_{record}"
    dump.mkdir()
    script = dump / "ptx_probe.py"
    script.write_text(_SM100_PTX_PROBE)
    proc = subprocess.run([sys.executable, str(script), str(dump), json.dumps(_SM100_STAGE3_RECORDS[record])], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"sm_100a trace-compile of the SM100 stage-3 {record} rendering failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = dict(ln.split(maxsplit=1) for ln in proc.stdout.splitlines() if ln.startswith(("PTX_MD5", "CONST")))
    # The module constants the record resolved to: no band, and B's head group exactly as the record spells it (1 on the
    # ten base records) -- a probe that silently defaulted the field would otherwise pin the wrong arm under a GQA name.
    bhg = _SM100_STAGE3_RECORDS[record].get("b_head_group", 1)
    assert out["CONST"] == f"causal_window 0 causal_diag True b_head_group {bhg}", f"the SM100 record rendered another arm: {out['CONST']}"
    got = out["PTX_MD5"].strip()
    print(f"\nSM100 stage-3 {record}: PTX md5 {got} (pre-edit record {want[record]} from {f.name})")
    assert got == want[record], f"{record}: PTX md5 {got} != the pre-edit record's {want[record]} ({f}) -- the SM100 chain's stage-3 rendering changed"


def test_stage3_b_head_group_default_folds_out_of_the_template():
    """``MatmulTemplateParams.b_head_group`` at its default 1 renders identically to a template without the parameter.  The
    rule, pinned on the template's SOURCE (no GPU): every code line of ``bprop_matmul_blackwell.py`` that reads the field is
    one of three -- the module-level ``getattr(PARAMS, "b_head_group", 1)`` (a record built before the field existed takes the
    default) and two ``cutlass.const_expr(b_head_group == 1)`` ternaries whose default arm is the untouched operand: ``_b_head``
    returns the decoded head ``tile_h`` itself (the ``//`` lives in the other arm only) and ``_host`` keeps B's descriptor head
    extent at ``n_head``.  So at 1 the traced body carries no division of the head index and no narrowed B extent -- the
    coordinate tuple and the descriptor are the ones that predated the field -- and ``_b_head`` is applied at exactly ONE site,
    B's head coordinate (A and C keep the decoded head).  A record WITHOUT the field (the dataclass minus ``b_head_group``)
    renders the same module constants as the default record.  The ``> 1`` arm is proven by the Rubin bitwise twin below."""
    import dataclasses

    from cudnn.frost.template_loader import DIGEST_GLOBAL, PARAMS_GLOBAL, load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, MatmulTemplateParams

    path = _sm100_kernel_path(_SM100_MATMUL_FILE)
    src = Path(path).read_text(encoding="utf-8")
    body = _code_only(src)
    # the field is read ONCE, at module level, through getattr with the default
    assert len(re.findall(r'^b_head_group = int\(getattr\(PARAMS, "b_head_group", 1\)\)$', src, flags=re.M)) == 1
    # every code line that reads it is that getattr or a const_expr(b_head_group == 1) ternary whose default arm is the operand itself
    reads = sorted(re.sub(r"\s+", " ", ln.strip()) for ln in body.splitlines() if "b_head_group" in ln)
    assert reads == sorted(
        [
            "b_head_group = int(getattr(PARAMS, , 1))",  # _code_only blanks the string literal
            "return tile_h if cutlass.const_expr(b_head_group == 1) else tile_h // cutlass.Int32(b_head_group)",
            "b_h = n_head if cutlass.const_expr(b_head_group == 1) else n_head // b_head_group",
        ]
    ), f"a use of b_head_group outside the const_expr fold reached the template: {reads}"
    # _b_head is a traced (@cute.jit) helper with that single return, applied at exactly one site: B's head coordinate
    assert "\n@cute.jit\ndef _b_head(" in body
    lines = body[body.index("def _b_head(") :].splitlines()
    end = next(i for i, ln in enumerate(lines[1:], 1) if ln and not ln[0].isspace())
    assert "\n".join(lines[:end]).count("return ") == 1
    assert body.count("_b_head(") == 2 and body.count("tile_h_b = _b_head(tile_h)") == 1, "the definition and B's coordinate site, nothing else"
    assert body.count("// b_head_group") == 1 and body.count("// cutlass.Int32(b_head_group)") == 1, "the two divisions sit in the > 1 arms only"
    # a record that never carried the field renders the same module constants as the default record
    record = MatmulTemplateParams(
        a_is_m_major=True, causal_mode=CAUSAL_K_HI, causal_gran=256, causal_shift=0, vec_bytes_epi=32, dtype_qkv=DTYPE_BF16, cgrp_tile_mn=(256, 256)
    )
    legacy_fields = [(f.name, f.type, dataclasses.field(default=f.default)) for f in dataclasses.fields(MatmulTemplateParams) if f.name != "b_head_group"]
    Legacy = dataclasses.make_dataclass("MatmulTemplateParamsBeforeBHeadGroup", legacy_fields, frozen=True)
    legacy = Legacy(**{name: getattr(record, name) for name, _, _ in legacy_fields})
    assert not hasattr(legacy, "b_head_group")
    default_mod, legacy_mod = (load_template(path, p, tag="test_sm107_stage3_default_fold") for p in (record, legacy))

    def consts(mod):
        return {
            k: v
            for k, v in vars(mod).items()
            if not k.startswith("__") and k not in (PARAMS_GLOBAL, DIGEST_GLOBAL) and isinstance(v, (bool, int, float, str, tuple, frozenset))
        }

    assert default_mod.b_head_group == legacy_mod.b_head_group == 1
    assert consts(default_mod) == consts(legacy_mod), "the default record and the record without the field render different module constants"


def test_stage3_cluster_tile_is_the_d256_one_on_the_rubin_line_only(monkeypatch):
    """The stage-3 cluster tile rule lives in the sm107 adapter: (256, 256) -- N = the head dim, no padding -- on the
    Rubin line (SM107-SM119) at d = 256, the SM100 chain's (512, 512) anywhere else (so the SM100 adapter, which never
    sets the field, renders the constants it always did).  ``STAGE3_D256_TILE = False`` is the bitwise pin's twin."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    for sm in (107, 110, 119):
        assert sm107._stage3_cgrp_tile_mn(sm, 256) == (256, 256), sm
    for sm, d in ((100, 256), (103, 256), (120, 256), (106, 256), (107, 512), (107, 128)):
        assert sm107._stage3_cgrp_tile_mn(sm, d) == (512, 512), (sm, d)
    assert sm107._stage3_cgrp_tile_mn(107, 256, d256_tile=False) == (512, 512)
    monkeypatch.setattr(sm107, "STAGE3_D256_TILE", False)
    assert sm107._stage3_cgrp_tile_mn(107, 256) == (512, 512), "the module constant is read at CALL time (the pin flips it)"
    assert sm107._stage3_cgrp_tile_mn(107, 256, d256_tile=True) == (256, 256)


def test_stage3_rejects_a_cluster_tile_the_template_has_no_row_for():
    from cudnn.sdpa.bwd.config_sm100 import STAGE3_CGRP_TILES, MatmulTemplateParams, validate_matmul_params

    assert (512, 512) in STAGE3_CGRP_TILES and (256, 256) in STAGE3_CGRP_TILES
    for bad in ((256, 512), (128, 256), (256,), (512, 128)):
        with pytest.raises(ValueError, match="cgrp_tile_mn"):
            validate_matmul_params(MatmulTemplateParams(cgrp_tile_mn=bad))
    validate_matmul_params(MatmulTemplateParams())  # the default row


def test_stage3_fp8_arm_records_are_validated_together():
    """The fp8 arm (``dtype_qkv=DTYPE_E4M3``) is admitted only as the rendering that was validated: the (256, 256) row, a
    descale or quantize epilogue (an undescaled fp8 accumulator has no consumer), an e4m3 output only under QUANT, and under
    THD the K64 arm's per-sequence trim (bottom-right spelled ``thd_causal_bottom_right`` on a trimmed causal mode, never a
    constant shift); and the bf16 / fp16 rows take no epilogue and no foreign output dtype.  Each raise names the reason."""
    import re

    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_LO, EPI_DESCALE, EPI_NONE, EPI_QUANT, MatmulTemplateParams, matmul_out_dtype, validate_matmul_params

    fp8 = dict(dtype_qkv=DTYPE_E4M3, cgrp_tile_mn=(256, 256))
    for ok in (
        dict(**fp8, epi_mode=EPI_DESCALE),
        dict(**fp8, epi_mode=EPI_QUANT, dtype_out=DTYPE_E4M3),
        dict(**fp8, epi_mode=EPI_QUANT, dtype_out=DTYPE_BF16),
        dict(**fp8, epi_mode=EPI_QUANT, dtype_out=DTYPE_FP16),
        dict(**fp8, epi_mode=EPI_QUANT),
        dict(**fp8, epi_mode=EPI_QUANT, thd_varlen=True, thd_rows_kv=True),  # the fp8 K64 arm's THD leg (the sm107 fp8 THD row)
        dict(**fp8, epi_mode=EPI_DESCALE, thd_varlen=True, thd_rows_kv=True),
        dict(**fp8, epi_mode=EPI_DESCALE, causal_mode=CAUSAL_K_LO, causal_gran=256, thd_varlen=True, thd_rows_kv=True, thd_causal_bottom_right=True),
    ):
        validate_matmul_params(MatmulTemplateParams(**ok))
    assert matmul_out_dtype(MatmulTemplateParams(**fp8, epi_mode=EPI_DESCALE)) == DTYPE_BF16, "the fp8 arm's inherited output is the bf16 true-unit value"
    assert matmul_out_dtype(MatmulTemplateParams(dtype_qkv=DTYPE_FP16)) == DTYPE_FP16
    for bad, needle in (
        (dict(dtype_qkv=DTYPE_E4M3, epi_mode=EPI_QUANT), "(256, 256) row only"),
        (dict(dtype_qkv=DTYPE_E4M3, cgrp_tile_mn=(512, 256), epi_mode=EPI_QUANT), "(256, 256) row only"),
        (dict(**fp8), "the fp8 arm requires one"),
        (dict(**fp8, epi_mode=EPI_NONE), "the fp8 arm requires one"),
        (dict(**fp8, epi_mode=EPI_DESCALE, dtype_out=DTYPE_E4M3), "needs EPI_QUANT"),
        (dict(**fp8, epi_mode=7), "epi_mode must be one of"),
        (dict(dtype_qkv=DTYPE_BF16, cgrp_tile_mn=(256, 256), epi_mode=EPI_QUANT), "belongs to the fp8 arm"),
        (dict(dtype_qkv=DTYPE_BF16, epi_mode=EPI_DESCALE), "belongs to the fp8 arm"),
        (dict(dtype_qkv=DTYPE_BF16, dtype_out=DTYPE_FP16), "store the io dtype"),
        (dict(dtype_qkv=DTYPE_BF16, dtype_out=DTYPE_E4M3), "needs EPI_QUANT"),
        (dict(dtype_qkv=1), "dtype_qkv must be"),
    ):
        with pytest.raises(ValueError, match=re.escape(needle)):
            validate_matmul_params(MatmulTemplateParams(**bad))


# Every module-level tile constant of the stage-3 rendering, per operand major.  The (512, 512) values ARE develop's
# (`origin/develop` @ 704b511e renders exactly these -- the upstream arch-100 rendering of
# CONFIG_sm100_256x256x128_128x256x32_cluster2x2_2ctamma); the (256, 256) values are the upstream arch-107 rendering
# of CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma with the fork's three pins (epi_n 64, 512 non-exclusive
# TMEM columns, no fallback cluster).  A change to either dict is a change to the rendered kernel: for the SM100 chain
# that is the byte-identity tripwire (its PTX must not move), for the d256 row it is the design (PERF_PARITY_DESIGN
# section A.3).
_STAGE3_ROW_INDEPENDENT = dict(
    mma_inst_shape_mnk=(256, 256, 16),
    cta_group=2,
    epi_tile_mn=(128, 64),
    threads_per_cta=256,
    multicast_b=False,
    b_mcast_slices=1,
    b_collector_ok=False,
    mma_size_n=1,
    mma_size_k=4,
    a_smem_m_step_bytes=16384,
    a_smem_desc_stride_byte_offset=1024,
    b_smem_desc_stride_byte_offset=1024,
    b_smem_desc_leading_byte_offset=8192,  # B is n-major on every stage-3 GEMM (D is contiguous and is N): `_MAJOR_CONSTS[True]`
    b_smem_k_step_bytes=2048,
    b_tma_group_elems=64,
    epi_n=64,
    epi_row_elems=64,
    epi_chunk_elems=64,
    epi_stage_rows=128,
    num_tmem_alloc_cols=512,
    tmem_alloc_exclusive=False,
    tile_swizzle_n=1,
    fallback_cluster_shape_mnk=None,
    mixed_b_pattern_pref=1,
    mixed_a_pattern_fb=1,
    mixed_b_pattern_fb=1,
    num_gemms=1,
    vec_bytes_epi=32,
)
_STAGE3_A_MAJOR = {  # a_is_m_major -> the `_MAJOR_CONSTS` of A
    False: dict(a_smem_desc_leading_byte_offset=16, a_smem_k_step_bytes=32, a_tma_group_elems=1),
    True: dict(a_smem_desc_leading_byte_offset=8192, a_smem_k_step_bytes=2048, a_tma_group_elems=64),
}
_STAGE3_ROWS = {
    (512, 512): dict(
        cgrp_tile_mnk=(512, 512, 64),
        cta_tile_mnk=(256, 128, 64),
        cluster_shape_mnk=(2, 2, 1),
        ab_stages=4,
        multicast_a=True,
        mma_size_m=2,
        acc_stages=1,
        mixed_a_pattern_pref=5,
        a_mcast={False: (2, False), True: (1, True)},  # a_is_m_major -> (a_mcast_slices, ab_empty_full_mask)
    ),
    (256, 256): dict(
        cgrp_tile_mnk=(256, 256, 64),
        cta_tile_mnk=(128, 128, 64),
        cluster_shape_mnk=(2, 1, 1),
        ab_stages=6,
        multicast_a=False,
        mma_size_m=1,
        acc_stages=2,
        mixed_a_pattern_pref=1,
        a_mcast={False: (1, False), True: (1, False)},
    ),
    (512, 256): dict(
        cgrp_tile_mnk=(512, 256, 64),
        cta_tile_mnk=(256, 128, 64),
        cluster_shape_mnk=(2, 1, 1),
        ab_stages=4,
        multicast_a=False,
        mma_size_m=2,
        acc_stages=1,
        mixed_a_pattern_pref=1,
        a_mcast={False: (1, False), True: (1, False)},
    ),
}
_SMEM_DESC_V0_LIMIT = 1 << 18  # a version-0 tcgen05 SMEM descriptor addresses the first 256 KiB
_SM100_OPTIN_SMEM = 227 * 1024  # the Blackwell per-CTA opt-in; every row must fit it (no oversized-mode dependency)


def _load_stage3(**params):
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm100 import MatmulTemplateParams

    return load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), MatmulTemplateParams(**params), tag="test_sm107_stage3")


@pytest.mark.parametrize("tile", sorted(_STAGE3_ROWS), ids=lambda t: f"{t[0]}x{t[1]}")
@pytest.mark.parametrize("a_is_m_major", (False, True), ids=("dK-Kmajor", "dQ-Mmajor"))
def test_stage3_tile_rows_render_their_upstream_constants(tile, a_is_m_major):
    """Each row of the template's tile-constants table renders EXACTLY the frozen constants above, with the SM100
    adapter's own records (default `cgrp_tile_mn`, both majors, dense and causal) hitting the (512, 512) dict -- the
    no-GPU byte-identity tripwire for develop's d512 chain (the PTX proof is in
    frost_dev/results/bwd_d256_sm107/perf/GEMM_D256.md).  Every row's SMEM layout (the template's declaration-order
    mirror) keeps both MMA-operand ring roots below 256 KiB -- the reach of the version-0 tcgen05 SMEM descriptor
    `Tcgen05SmemDesc.build()` emits (rules/mma-tma-matrix.md s6) -- and fits the 227 KiB opt-in budget."""
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16

    row = _STAGE3_ROWS[tile]
    causal_mode = CAUSAL_K_HI if a_is_m_major else CAUSAL_K_LO  # the sm107 chain's pairing; the modes do not touch the tile constants
    params = dict(a_is_m_major=a_is_m_major, b_is_n_major=True, causal_gran=256, causal_shift=0, vec_bytes_epi=32, dtype_qkv=DTYPE_BF16)
    mods = [_load_stage3(causal_mode=causal_mode, cgrp_tile_mn=tile, **params), _load_stage3(causal_mode=CAUSAL_K_NONE, cgrp_tile_mn=tile, **params)]
    if tile == (512, 512):
        # The SM100 adapter's exact spelling: no `cgrp_tile_mn` at all (api_dsl.py `SdpaBwdDslSm100.compile`).
        mods.append(_load_stage3(causal_mode=causal_mode, thd_varlen=False, **params))
    expect = dict(_STAGE3_ROW_INDEPENDENT, **_STAGE3_A_MAJOR[a_is_m_major], **{k: v for k, v in row.items() if k != "a_mcast"})
    expect["a_mcast_slices"], expect["ab_empty_full_mask"] = row["a_mcast"][a_is_m_major]
    for mod in mods:
        got = {name: getattr(mod, name) for name in expect}
        assert got == expect, {k: (got[k], expect[k]) for k in expect if got[k] != expect[k]}
        assert mod._ROW.config.startswith("CONFIG_sm100_") and mod._TILE_ROWS[tile] is mod._ROW
        layout = mod._smem_layout_bytes()
        print(f"\nstage-3 {tile} a_is_m_major={a_is_m_major}: SMEM layout {layout}")
        assert layout["smem_a_0"] < _SMEM_DESC_V0_LIMIT and layout["smem_b_0"] < _SMEM_DESC_V0_LIMIT, layout
        assert layout["total"] <= _SM100_OPTIN_SMEM, layout
        assert layout["total"] == 231424, "every row spends the same 226 KiB: stages x (A + B) + 32 KiB staging + 2 KiB of barriers"
        assert layout["smem_d"] + 2 * 128 * 64 * 2 == layout["total"]


# The fp8 arm's constants: the upstream arch-107 rendering of CONFIG_sm100_128x256x128_128x256x64_cluster2x1_2ctamma at
# e4m3 operands (the (256, 256) row at 1-byte operands), with the fork's pins (epi_n 64 -> a 64-B staging row at e4m3 out).
_STAGE3_FP8_ARM = dict(
    cgrp_tile_mnk=(256, 256, 128),
    cta_tile_mnk=(128, 128, 128),
    mma_inst_shape_mnk=(256, 256, 64),
    mma_k_dim=1,
    mma_size_k=2,
    mma_size_m=1,
    ab_stages=6,
    acc_stages=2,
    cluster_shape_mnk=(2, 1, 1),
    b_smem_desc_leading_byte_offset=16384,  # B n-major at 1 B/elem: 128 e4m3 per 128-B swizzle row
    b_smem_k_step_bytes=8192,  # 64 K rows x 128 B per MMA k-block
    b_tma_group_elems=128,
)
_STAGE3_FP8_A_MAJOR = {
    False: dict(a_smem_desc_leading_byte_offset=16, a_smem_k_step_bytes=64, a_tma_group_elems=1),  # dK: dS[kv, q] K-major, 64 B per K64 block
    True: dict(a_smem_desc_leading_byte_offset=16384, a_smem_k_step_bytes=8192, a_tma_group_elems=128),  # dQ: dS^T[q, kv] M-major
}


@pytest.mark.parametrize("a_is_m_major", (False, True), ids=("dK-Kmajor", "dQ-Mmajor"))
@pytest.mark.parametrize("epi", ("descale", "quant-e4m3", "quant-bf16"))
def test_stage3_fp8_arm_renders_the_upstream_k64_constants(a_is_m_major, epi):
    """The fp8 arm renders Rubin's dense-FP8 K64 form (256x256x64, idesc k_dim=1, F8F6F4; 128 e4m3 per K stage in the same
    128-B swizzle row, so a stage is TWO MMA k-blocks) with the (256, 256) row's tile and the 1-byte operand-major constants
    lifted from the upstream rendering; its epilogue staging row follows the OUTPUT dtype (128 B / Swizzle(3,4,3) / s128b at
    bf16, 64 B / Swizzle(2,4,3) / s64b at e4m3 -- store swizzle and store-descriptor swizzle move together), both operand
    ring roots stay under the version-0 descriptor window and the layout fits the 227 KiB opt-in."""
    import cutlass
    import cutlass.experimental.cuda.tensor_map as _tma
    import cutlass.experimental.primitives as nvvm

    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, EPI_DESCALE, EPI_QUANT

    mode, out = {"descale": (EPI_DESCALE, -1), "quant-e4m3": (EPI_QUANT, DTYPE_E4M3), "quant-bf16": (EPI_QUANT, DTYPE_BF16)}[epi]
    mod = _load_stage3(
        a_is_m_major=a_is_m_major,
        b_is_n_major=True,
        causal_mode=CAUSAL_K_HI if a_is_m_major else CAUSAL_K_LO,
        causal_gran=256,
        causal_shift=0,
        vec_bytes_epi=32,
        dtype_qkv=DTYPE_E4M3,
        cgrp_tile_mn=(256, 256),
        epi_mode=mode,
        dtype_out=out,
    )
    expect = dict(_STAGE3_FP8_ARM, **_STAGE3_FP8_A_MAJOR[a_is_m_major])
    got = {name: getattr(mod, name) for name in expect}
    assert got == expect, {k: (got[k], expect[k]) for k in expect if got[k] != expect[k]}
    assert mod._TILE_ROWS[(256, 256)] is mod._ROW and mod._IS_FP8 and mod.epi_mode == mode
    assert mod.ab_dtype is cutlass.Float8E4M3FN and mod.mma_kind == nvvm.Tcgen05MMAKind.F8F6F4
    assert mod.cd_dtype is (cutlass.Float8E4M3FN if epi == "quant-e4m3" else cutlass.BFloat16)
    if epi == "quant-e4m3":
        assert (mod._EPI_ROW_BYTES, mod._EPI_TMA_SWIZZLE) == (64, _tma.TensorMapSwizzle.s64b) and str(mod._EPI_SWIZZLE) == str(cutlass.Swizzle(2, 4, 3))
    else:
        assert (mod._EPI_ROW_BYTES, mod._EPI_TMA_SWIZZLE) == (128, _tma.TensorMapSwizzle.s128b) and str(mod._EPI_SWIZZLE) == str(cutlass.Swizzle(3, 4, 3))
    # ... and the bf16 rows keep the F16 form untouched by the arm's existence.
    bf16 = _load_stage3(
        a_is_m_major=a_is_m_major, b_is_n_major=True, causal_gran=256, causal_shift=0, vec_bytes_epi=32, dtype_qkv=DTYPE_BF16, cgrp_tile_mn=(256, 256)
    )
    assert (bf16.mma_inst_shape_mnk, bf16.mma_k_dim, bf16.mma_size_k, bf16.mma_kind, bf16.epi_mode) == ((256, 256, 16), 0, 4, nvvm.Tcgen05MMAKind.F16, 0)
    assert (bf16._EPI_ROW_BYTES, bf16._EPI_TMA_SWIZZLE) == (128, _tma.TensorMapSwizzle.s128b)
    layout = mod._smem_layout_bytes()
    print(f"\nstage-3 fp8 {epi} a_is_m_major={a_is_m_major}: SMEM layout {layout}")
    assert layout["smem_a_0"] < _SMEM_DESC_V0_LIMIT and layout["smem_b_0"] < _SMEM_DESC_V0_LIMIT, layout
    assert layout["total"] <= _SM100_OPTIN_SMEM, layout
    # 6 x (16 + 16) KiB of e4m3 operands = the bf16 rows' footprint; the e4m3-out staging is 2 x 8 KiB instead of 2 x 16.
    assert layout["total"] == (215040 if epi == "quant-e4m3" else 231424), layout


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(dt=torch.float16),
        dict(hq=8, hkv=2, sq=256, skv=256),
        dict(sq=500, skv=500, is_causal=True, window_size_left=256),
        dict(b=1, hq=4, hkv=2, sq=512, skv=1024, is_causal=True, causal_bottom_right=True),
        dict(b=1, hq=2, sq=500, skv=1024, is_causal=True, causal_bottom_right=True),
        dict(sq=257, skv=129),
    ],
    ids=["dense", "fp16", "gqa", "swa-padded", "br-rect", "br-ragged-sq", "257x129"],
)
def test_half_adapter_backstop_admits_the_served_matrix_and_sizes_its_workspace_at_build(kw):
    """The adapter's check_support admits every graph the row claims, and its workspace is a pure function of the
    compile geometry: the carve plan's aligned sum, identical on every call, every planned buffer 128-B aligned."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107
    from cudnn.sdpa.fwd.api_dsl import ws_align

    api = _adapter(SdpaBwdDslSm107, **kw)
    assert api.check_support()
    plan = api._scratch_plan()
    names = [n for n, _n, _d in plan]
    assert names[:4] == ["delta", "ds_ws", "seq_kv", "desc_words"] and len(set(names)) == len(names)
    total = api.scratch_workspace_bytes()
    assert total == api.scratch_workspace_bytes() == sum(ws_align(n * d.itemsize) for _, n, d in plan) > 0
    assert ("q_pad" in names) == (api.s_q_max % 128 != 0) and ("k_pad" in names) == (api.s_k_max % 256 != 0)
    assert ("dk_part" in names) == (api.h_q != api.h_kv)
    from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_WS_BUDGET_BYTES

    assert api._b_chunk * api._qh_chunk * api._skv_pad * api._sq_pad * 2 <= _SM107_WS_BUDGET_BYTES or api._qh_chunk == api._gqa_group


@pytest.mark.parametrize(
    "kw, needle",
    [
        (dict(dt=torch.float32), "dtype"),
        (dict(hq=6, hkv=4), "multiple of h_kv"),
        (dict(sq=1), "decode"),
        (dict(is_causal=True, window_size_right=16), "right-band"),
        (dict(window_size_left=0), "window_left > 0"),
        (dict(causal_bottom_right=True), "requires a causal mask"),
        (dict(deterministic=True), "deterministic"),
        (dict(seq_q_lens_present=True), "seq_q_lens"),
        (dict(seq_kv_lens_present=True, seq_q_lens_present=True), "seq_q_lens"),
        (dict(thd=True), "max_total_seq_len"),
        (dict(thd=True, max_total_seq_len_q=1024, max_total_seq_len_kv=1024, seq_kv_lens_present=True), "mutually exclusive"),
    ],
    ids=[
        "fp32",
        "gqa-ratio",
        "decode",
        "right-band",
        "swa-zero",
        "br-without-causal",
        "deterministic",
        "seq-q-lens",
        "seq-q-and-kv-lens",
        "thd-undeclared-totals",
        "thd-and-kv-lens",
    ],
)
def test_half_adapter_backstop_refuses_what_the_row_declines(kw, needle):
    """Reaching one of these raises means the row lied; each is a ValueError naming the reason, never an assert."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    with pytest.raises(ValueError, match=needle):
        _adapter(SdpaBwdDslSm107, **kw).check_support()


def test_half_adapter_admits_thd_and_sizes_its_packed_workspace_at_build():
    """The half row's THD plan: declared totals tighten the token capacity (a MIN), the dS workspace is kv-BLOCKED at the
    kernel's 256-row block with every sequence padded to it, no staging and no batch chunking, the metadata region carries the
    main kernel's (5 + B) tensor maps, the GQA partials ride the packed kv capacity, and the two length operands join the launch record
    (``length_form``) at slots 9 / 10, ahead of the appended standalone-only delta slot (``test_sdpa_bwd_thd_sm107.py`` holds its
    contract)."""
    from cudnn.frost.tile_dsl.thd import THD_BWD_MAPS_META_WORDS
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107
    from cudnn.sdpa.bwd.prepared_sm107 import ATTRIBUTES_F16_THD, EXTERNAL_DELTA_ROLE, ROLES_F16_THD

    api = _adapter(SdpaBwdDslSm107, b=3, hq=4, hkv=2, sq=300, skv=500, thd=True, max_total_seq_len_q=628, max_total_seq_len_kv=5000)
    assert api.check_support()
    assert (api._t_q_cap, api._t_kv_cap) == (628, 1500), "the declared total tightens the capacity, never widens it (B * S_max = 1500 on the kv side)"
    assert api._ws_rows_cap == -(-(1500 + 3 * 256) // 256) * 256 and api._sq_pad == 384 and not api._q_padded and not api._kv_padded
    assert (api._b_chunk, api._qh_chunk) == (1, 4), "one packed batch; the head chunk divides a per-head slab of R_kv_cap x S_q_pad"
    plan = {n: shape for n, shape, _d in api._scratch_shapes()}
    assert plan["delta"] == (1, 4, 640) and plan["ds_ws"] == (1, 4, api._ws_rows_cap, 384)
    assert plan["seq_kv"] == (THD_BWD_MAPS_META_WORDS(3, 8),) and plan["desc_words"] == (4 * 16,)
    assert plan["dv_part"] == (1, 1500, 4, 256) and plan["dk_part"] == (1, 1500, 4, 256) and "q_pad" not in plan and "k_pad" not in plan
    assert api._template_params().thd_varlen and not api._template_params().seq_kv_lens_present
    assert ROLES_F16_THD[9:11] == ("seq_q", "seq_kv") and ATTRIBUTES_F16_THD[9:11] == ("seq_len_q", "seq_len_kv") and len(ROLES_F16_THD) == 12
    assert ROLES_F16_THD[-1] == ATTRIBUTES_F16_THD[-1] == EXTERNAL_DELTA_ROLE, "the delta slot is appended LAST, after the two lengths"
    # The stage-3 records under THD: the THD arm on, rows KV-major, the SAME two-sided trim as the dense records (per sequence
    # in the template: a constant shift of 0, bottom-right spelled as `thd_causal_bottom_right`), dQ once per head chunk.
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, validate_matmul_params

    mod = types.SimpleNamespace(CFG=types.SimpleNamespace(TILE_M=128, CTA_MMA=2))
    for causal, bottom_right, left in (
        (False, False, None),
        (True, False, None),
        (True, True, None),
        (True, False, 200),
        (True, True, 64),
        (False, False, 200),
    ):
        kw = dict(is_causal=causal, causal_bottom_right=bottom_right)
        if left is not None:
            kw["window_size_left"] = left - 1
        api_c = _adapter(SdpaBwdDslSm107, b=3, hq=4, hkv=2, sq=300, skv=500, thd=True, max_total_seq_len_q=628, max_total_seq_len_kv=1500, **kw)
        dk, dq = api_c._stage3_records(mod, (256, 256))
        validate_matmul_params(dk)
        validate_matmul_params(dq)
        assert dk.thd_varlen and dq.thd_varlen and dk.thd_rows_kv and dq.thd_rows_kv
        band = causal or left is not None
        assert (dk.causal_mode, dq.causal_mode) == ((CAUSAL_K_LO, CAUSAL_K_HI) if band else (CAUSAL_K_NONE, CAUSAL_K_NONE)), (causal, bottom_right, left)
        assert dk.causal_shift == dq.causal_shift == 0, "the THD shift is per sequence, never the envelope's"
        assert dk.thd_causal_bottom_right == dq.thd_causal_bottom_right == (causal and bottom_right)
        assert dk.causal_window == dq.causal_window == (left - 1 if left is not None else 0) and dk.causal_diag == dq.causal_diag == (causal or not band)
        assert dq.b_head_group == 2 and dk.b_head_group == 1 and not dk.a_is_m_major and dq.a_is_m_major
    # A padded token stride is admitted (what mismatch() admits for a THD row without thd_head_stride); a non-D head stride is not.
    from cudnn.api_base import TensorDesc

    def _packed(shape, stride, name):
        return TensorDesc(
            dtype=torch.bfloat16,
            shape=shape,
            stride=stride,
            stride_order=TensorDesc._compute_stride_order(shape, stride),
            device=torch.device("cuda", 0),
            name=name,
        )

    wide = _adapter(SdpaBwdDslSm107, b=2, hq=2, sq=256, skv=256, thd=True, max_total_seq_len_q=512, max_total_seq_len_kv=512)
    wide.q_desc = _packed((2, 2, 256, 256), (256 * 1024, 256, 1024, 1), "q")  # token stride 2 * H * D
    assert wide.check_support()
    bad = _adapter(SdpaBwdDslSm107, b=2, hq=2, sq=256, skv=256, thd=True, max_total_seq_len_q=512, max_total_seq_len_kv=512)
    bad.q_desc = _packed((2, 2, 256, 256), (256 * 1024, 512, 1024, 1), "q")  # head stride 2 * D
    with pytest.raises(ValueError, match="packed BSHD rows"):
        bad.check_support()


def test_half_adapter_admits_per_batch_kv_lengths():
    """The standalone surface of the half row serves the caller's per-batch kv lengths (``seq_kv_lens_present=True`` at
    construction, ``execute(seq_kv_lens=)``): the body's padded-mask arm (``seq_kv_lens_present`` on the template record, the
    same arm a ragged S_kv selects) reads ``seq_kv_lens[b]`` in place of the uniform fill.  Host pins: the arm is selected by
    the lengths alone (a tile-multiple S_kv), the workspace plan is unchanged (the ``seq_kv`` region stays carved -- fixed
    ABI), the prepared spec carries the lengths as its tenth operand, and the stage-3 zero-fill returns for exactly the
    bottom-right arms (the per-batch diagonal moves the kernel's band under the GEMMs' uniform trim) and nothing else."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107, _stage3_needs_zero_fill, _stage3_trim_window
    from cudnn.sdpa.bwd.prepared_sm107 import ATTRIBUTES_F16, ROLES_F16

    api = _adapter(SdpaBwdDslSm107, seq_kv_lens_present=True)
    assert api.check_support()
    assert api._template_params().seq_kv_lens_present and not api._kv_padded, "the arm is selected by the caller's lengths alone"
    assert [n for n, _n, _d in api._scratch_plan()] == [
        n for n, _n, _d in _adapter(SdpaBwdDslSm107)._scratch_plan()
    ], "no new scratch: the lengths are the caller's"
    ragged = _adapter(SdpaBwdDslSm107, skv=1000, seq_kv_lens_present=True)
    assert ragged.check_support() and ragged._template_params().seq_kv_lens_present and ragged._kv_padded
    assert "k_pad" in [n for n, _n, _d in ragged._scratch_plan()], "a ragged S_kv keeps its zero-filled K / V staging under per-batch lengths"
    for kw in (dict(is_causal=True), dict(is_causal=True, causal_bottom_right=True), dict(is_causal=True, window_size_left=199), dict(hq=4, hkv=2)):
        assert _adapter(SdpaBwdDslSm107, seq_kv_lens_present=True, **kw).check_support(), kw
    # Slot 9 is the lengths (a graph attribute); slot 10, appended after it, is the standalone-only external delta.
    assert ROLES_F16[9] == "seq_kv" and ATTRIBUTES_F16[9] == "seq_len_kv" and len(ROLES_F16) == len(ATTRIBUTES_F16) == 11
    # The zero-fill rule under per-batch lengths: bottom-right only (with or without a window); top-left bands and dense never.
    assert _stage3_needs_zero_fill(True, None, True, 512, 1024, 256, per_batch_kv=True)
    assert _stage3_needs_zero_fill(True, 199, True, 512, 1024, 256, per_batch_kv=True)
    assert not _stage3_needs_zero_fill(True, None, False, 512, 1024, 256, per_batch_kv=True)
    assert not _stage3_needs_zero_fill(True, 199, False, 1024, 1024, 256, per_batch_kv=True)
    assert not _stage3_needs_zero_fill(False, 199, False, 1024, 1024, 256, per_batch_kv=True)
    assert not _stage3_needs_zero_fill(False, None, False, 512, 1024, 256, per_batch_kv=True)
    assert not _stage3_needs_zero_fill(True, None, True, 512, 1024, 256), "uniform lengths: the two-sided trim reads only what was written (unchanged)"
    # The stage-3 trim's window under per-batch lengths: dropped for bottom-right (a window edge anchored on the uniform
    # diagonal would skip a shorter batch's live tiles), kept for top-left bands and for every uniform-length graph.
    assert _stage3_trim_window(199, True, True, True) is None
    assert _stage3_trim_window(None, True, True, True) is None
    assert _stage3_trim_window(199, True, False, True) == 199
    assert _stage3_trim_window(199, False, False, True) == 199
    assert _stage3_trim_window(199, True, True, False) == 199
    # ... and the half row's records take it: bottom-right + window + per-batch lengths render the plain bottom-right band
    # (`causal_window == 0`, the diagonal kept) where the uniform-length twin keeps the window.
    mod = types.SimpleNamespace(CFG=types.SimpleNamespace(TILE_M=128, CTA_MMA=2))
    br_window = dict(sq=512, skv=1024, is_causal=True, causal_bottom_right=True, window_size_left=199)
    dk, dq = _adapter(SdpaBwdDslSm107, seq_kv_lens_present=True, **br_window)._stage3_records(mod, (256, 256))
    assert (dk.causal_window, dq.causal_window) == (0, 0) and dk.causal_diag and dq.causal_diag and dk.causal_shift == dq.causal_shift == 512
    dk, dq = _adapter(SdpaBwdDslSm107, **br_window)._stage3_records(mod, (256, 256))
    assert (dk.causal_window, dq.causal_window) == (199, 199) and dk.causal_shift == dq.causal_shift == 512
    dk, dq = _adapter(SdpaBwdDslSm107, seq_kv_lens_present=True, sq=512, skv=1024, is_causal=True, window_size_left=199)._stage3_records(mod, (256, 256))
    assert (dk.causal_window, dq.causal_window) == (199, 199) and dk.causal_shift == dq.causal_shift == 0, "a top-left band keeps its window"


def test_half_adapter_execute_requires_the_lengths_exactly_when_planned(monkeypatch):
    """The lengths operand is a PLAN fact (the prepared spec binds it exactly when ``seq_kv_lens_present``): execute refuses a
    missing buffer on a plan built with it, an unrequested one on a plan built without, and ``seq_q_lens`` always -- each a
    ValueError naming the reason, before anything compiles or launches."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    lens = torch.zeros(2, dtype=torch.int32)
    with_lens = _adapter(SdpaBwdDslSm107, seq_kv_lens_present=True)
    monkeypatch.setattr(with_lens, "compile", lambda: pytest.fail("refused before compile"))
    with pytest.raises(ValueError, match="exactly when"):
        with_lens.execute(*([None] * 9))
    plain = _adapter(SdpaBwdDslSm107)
    monkeypatch.setattr(plain, "compile", lambda: pytest.fail("refused before compile"))
    with pytest.raises(ValueError, match="exactly when"):
        plain.execute(*([None] * 9), seq_kv_lens=lens)
    with pytest.raises(ValueError, match="seq_q_lens"):
        plain.execute(*([None] * 9), seq_q_lens=lens)
    with pytest.raises(ValueError, match="seq_q_lens"):
        with_lens.execute(*([None] * 9), seq_kv_lens=lens, seq_q_lens=lens)


def test_fp8_adapter_backstop_and_workspace(monkeypatch):
    """The fp8 row's twin: E4M3 payloads with e4m3 / bf16 / fp16 gradients pass; E5M2, a mixed gradient triple, a half
    O payload and an off-contract gradient dtype raise.  Its workspace (the shipped e4m3 dS): the e4m3 dS chunk, the bf16
    dV partials, bf16 dK partials under GQA only, the amax scratch -- no Q / K upcasts, no dQ / dK scratch (the GEMM
    epilogue quantizes them in place); the bf16-dS twin (``FP8_DS_DTYPE = DTYPE_BF16``) adds the bf16 Q / K copies and the
    three bf16 partials, at twice the dS bytes.  Neither chunks the batch."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8
    from cudnn.sdpa.bwd.config_sm100 import EPI_DESCALE, EPI_QUANT

    e4m3 = torch.float8_e4m3fn
    assert sm107.FP8_DS_DTYPE == DTYPE_E4M3, "the e4m3 dS chain is what ships"
    for grad in (e4m3, torch.bfloat16, torch.float16):
        api = _adapter(SdpaBwdDslSm107Fp8, b=2, hq=8, hkv=2, sq=257, skv=129, dt=e4m3, grad_dt=grad, is_causal=True)
        assert api.check_support()
        names = [n for n, _n, _d in api._scratch_plan()]
        for n in ("dv_part", "dk_part", "amax_scratch", "q_pad", "k_pad", "lse_pad"):
            assert n in names, n
        for n in ("dq_ws", "q_bf16", "k_bf16"):
            assert n not in names, f"{n}: the e4m3 chain reads the e4m3 payloads directly and quantizes dQ / dK in the GEMM epilogue"
        assert api._ds_dtype == e4m3 and api._bpe_ds == 1
        assert {n: (sh, dt) for n, sh, dt in api._scratch_shapes()}["ds_ws"] == ((2, 8, 256, 384), e4m3), "the dS chunk is e4m3 at the padded extents"
        assert api._b_chunk == 2, "the fp8 body has no batch_base: the whole batch is in-grid"
        assert api.scratch_workspace_bytes() == api.scratch_workspace_bytes() > 0
        assert api._template_params().dtype_ds == DTYPE_E4M3
    # MHA: the GEMM writes the caller's dK straight -- no dk_part either.
    mha = _adapter(SdpaBwdDslSm107Fp8, b=1, hq=2, hkv=2, sq=512, skv=512, dt=e4m3, grad_dt=e4m3)
    assert mha.check_support() and [n for n, _n, _d in mha._scratch_plan()] == ["delta", "ds_ws", "seq_kv", "desc_words", "dv_part", "amax_scratch"]
    e4m3_bytes = mha.scratch_workspace_bytes()
    # The bf16-dS twin: today's chain end to end.
    monkeypatch.setattr(sm107, "FP8_DS_DTYPE", DTYPE_BF16)
    twin = _adapter(SdpaBwdDslSm107Fp8, b=1, hq=2, hkv=2, sq=512, skv=512, dt=e4m3, grad_dt=e4m3)
    assert twin.check_support() and twin._ds_dtype == torch.bfloat16 and twin._bpe_ds == 2
    names = [n for n, _n, _d in twin._scratch_plan()]
    assert names == ["delta", "ds_ws", "seq_kv", "desc_words", "dv_part", "dk_part", "dq_ws", "q_bf16", "k_bf16", "amax_scratch"]
    assert twin._template_params().dtype_ds == DTYPE_BF16 and twin.scratch_workspace_bytes() > e4m3_bytes
    assert twin._qh_chunk <= mha._qh_chunk, "the e4m3 dS chunk doubles the heads per launch at the same budget"
    monkeypatch.setattr(sm107, "FP8_DS_DTYPE", DTYPE_E4M3)
    with pytest.raises(ValueError, match="not served"):
        _adapter(SdpaBwdDslSm107Fp8, dt=torch.float8_e5m2, grad_dt=e4m3).check_support()
    with pytest.raises(ValueError, match="not in"):
        _adapter(SdpaBwdDslSm107Fp8, dt=e4m3, grad_dt=torch.float32).check_support()
    api = _adapter(SdpaBwdDslSm107Fp8, dt=e4m3, grad_dt=e4m3)
    api.dk_desc = _adapter_desc((2, 2, 512, _D), torch.bfloat16, "dK")
    with pytest.raises(ValueError, match="share one dtype"):
        api.check_support()
    api = _adapter(SdpaBwdDslSm107Fp8, dt=e4m3, grad_dt=e4m3)
    api.o_desc = _adapter_desc((2, 2, 512, _D), torch.bfloat16, "o")
    with pytest.raises(ValueError, match="FP8 payload"):
        api.check_support()


def test_half_adapter_external_delta_is_a_plan_fact_that_drops_the_region(monkeypatch):
    """``external_delta=True`` (appended, default off) declares that the caller computes stage 1's delta: the carve loses its
    ``delta`` region (exactly ``ws_align(B * H_q * S_q_pad * 4)`` bytes), the contract shape is ``external_delta_shape`` =
    ``(B, H_q, S_q_pad)`` on both plans, Capabilities and the rest of the plan are untouched; the fp8 and mxfp8 rows decline
    the flag typed (their delta is the dot of their own payloads -- the descaled fp8 dot of the scaled pre-pass, the o_f16 /
    dO_f16 ports' dot -- and the mxfp8 row inherits ``__init__`` / ``_scratch_shapes`` from the half row, so without its own
    decline the flag would drop the ``delta`` region its host still carves and die untyped at trace).  The carve under BOTH
    appended plan facts is pinned here too: ``seq_kv_lens_present`` keeps the ``seq_kv`` region carved (fixed ABI, no new
    scratch) and ``external_delta`` drops only ``delta``, so the combined plan is exactly the delta's bytes smaller than the
    lengths-only plan and carves what the delta-only plan carves."""
    from test_sdpa_bwd_mxfp8_sm107 import _RUBIN_CC, _mxfp8_adapter

    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107, SdpaBwdDslSm107Fp8
    from cudnn.sdpa.fwd.api_dsl import ws_align

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != _RUBIN_CC:
        # The mxfp8 row's shipped dS policy (P-b) declines typed off the Rubin line at check_support (the stage-3 arm's 576-column
        # exclusive TMEM); this is a host-side PLAN pin, so the prepared plan's device query answers SM107 as in the mxfp8 suite.
        monkeypatch.setattr(prepared_sm107, "_sm", lambda api: 107)
    own = _adapter(SdpaBwdDslSm107, hq=8, hkv=2, sq=500, skv=500, is_causal=True)
    ext = _adapter(SdpaBwdDslSm107, hq=8, hkv=2, sq=500, skv=500, is_causal=True, external_delta=True)
    assert own.check_support() and ext.check_support()
    assert own.external_delta is False and ext.external_delta is True
    names, ext_names = ([n for n, _n, _d in api._scratch_plan()] for api in (own, ext))
    assert names[0] == "delta" and "delta" not in ext_names and ext_names == names[1:]
    assert own.external_delta_shape == ext.external_delta_shape == (2, 8, 512), "S_q_pad = S_q rounded up to the 128-row q tile"
    assert own.scratch_workspace_bytes() - ext.scratch_workspace_bytes() == ws_align(2 * 8 * 512 * 4)
    assert (own._b_chunk, own._qh_chunk, own._sq_pad, own._skv_pad) == (ext._b_chunk, ext._qh_chunk, ext._sq_pad, ext._skv_pad)
    e4m3 = torch.float8_e4m3fn
    # The quantized rows take the flag too (the dense plans here; their THD plans serve it as well -- test_sdpa_bwd_thd_*_sm107.py):
    # the carve drops ``delta`` under it and the standalone-only role is appended LAST on both role lists (after the per-batch kv
    # lengths).  The fp8 kernel reads delta in
    # TRUE units unscaled, so a caller's bf16-derived delta binds AS IS; the mxfp8 delta is the dot of the f16 ports.
    for api_ext, api_own in (
        (_adapter(SdpaBwdDslSm107Fp8, dt=e4m3, grad_dt=e4m3, external_delta=True), _adapter(SdpaBwdDslSm107Fp8, dt=e4m3, grad_dt=e4m3)),
        (_mxfp8_adapter(external_delta=True), _mxfp8_adapter()),
    ):
        assert api_ext.check_support() and api_own.check_support()
        assert api_ext.external_delta is True and api_own.external_delta is False
        q_own_names, q_ext_names = ([n for n, _n, _d in api._scratch_plan()] for api in (api_own, api_ext))  # not the half row's names
        assert q_own_names[0] == "delta" and "delta" not in q_ext_names and q_ext_names == q_own_names[1:], api_ext._NAME
        assert api_own.scratch_workspace_bytes() - api_ext.scratch_workspace_bytes() == ws_align(math.prod(api_own.external_delta_shape) * 4)
    # Both plan facts at once (the compiled twin is the Rubin test_adapter_per_batch_kv_lengths_compose_with_the_gate_kernels_external_delta)
    lengths_only = _adapter(SdpaBwdDslSm107, hq=8, hkv=2, sq=500, skv=500, is_causal=True, seq_kv_lens_present=True)
    both = _adapter(SdpaBwdDslSm107, hq=8, hkv=2, sq=500, skv=500, is_causal=True, seq_kv_lens_present=True, external_delta=True)
    assert lengths_only.check_support() and both.check_support()
    assert (lengths_only.seq_kv_lens_present, lengths_only.external_delta, both.seq_kv_lens_present, both.external_delta) == (True, False, True, True)
    both_names = [n for n, _n, _d in both._scratch_plan()]
    assert [n for n, _n, _d in lengths_only._scratch_plan()] == names, "the lengths add no scratch: the seq_kv region stays carved"
    assert "seq_kv" in both_names and "delta" not in both_names and both_names == ext_names, "the combined carve is the delta-only carve"
    assert lengths_only.scratch_workspace_bytes() == own.scratch_workspace_bytes()
    assert lengths_only.scratch_workspace_bytes() - both.scratch_workspace_bytes() == ws_align(2 * 8 * 512 * 4)
    assert both.external_delta_shape == (2, 8, 512)


def test_prepared_bwd_launch_frames_only_the_declared_standalone_only_roles_as_absent():
    """The graph binder (``prepared.PreparedBwdLaunch``) reads every spec attribute STRICTLY off the ``SdpaBinding`` -- a
    misspelled role in a ``BwdLaunchSpec`` still fails at plan build, never as a silent None -- and frames as absent exactly the
    roles listed in ``BwdLaunchSpec.standalone_only_roles``: the half row's caller-provided ``delta`` (``prepared_sm107.EXTERNAL_DELTA_ROLE``),
    which no graph declares and no other sm107 role list carries.  Host-only (fake binding, no artifact)."""
    from dataclasses import replace
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.prepared import BwdLaunchSpec, PreparedBwdLaunch

    class _Bound:
        def __init__(self, uid, dim, stride):
            self.uid, self.dim, self.stride = uid, dim, stride

        def get_uid(self):
            return self.uid

        def get_dim(self):
            return self.dim

        def get_stride(self):
            return self.stride

    delta = prepared_sm107.EXTERNAL_DELTA_ROLE
    assert delta == "delta" and prepared_sm107.ROLES_F16[-1] == prepared_sm107.ATTRIBUTES_F16[-1] == delta
    # Every dense sm107 row carries the two appended slots in the same order: the per-batch kv lengths, then the standalone-only delta.
    for roles, attributes in (
        (prepared_sm107.ROLES_FP8, prepared_sm107.ATTRIBUTES_FP8),
        (prepared_sm107.ROLES_MXFP8, prepared_sm107.ATTRIBUTES_MXFP8),
    ):
        assert roles[-2:] == ("seq_kv", delta) and attributes[-2:] == ("seq_len_kv", delta) and len(roles) == len(attributes), roles[-2:]
    binding = SimpleNamespace(q=_Bound(11, (1, 2, 512, 256), (262144, 256, 512, 1)), stats=_Bound(12, (1, 2, 512, 1), (1024, 512, 1, 1)))
    base = dict(artifact=None, fn=None, operands=(), workspace_bytes=0, device_index=0, scale=1.0, name="probe")
    spec = BwdLaunchSpec(**base, roles=("q", "stats", delta), attributes=("q", "stats", delta), standalone_only_roles=(delta,), native_binding=False)
    launch = PreparedBwdLaunch(spec, binding)
    assert launch._roles == ["q", "stats"] and launch._uids == [11, 12]
    assert launch._geometry[:2] == (((1, 2, 512, 256), (262144, 256, 512, 1)), ((1, 2, 512, 1), (1024, 512, 1, 1))) and launch._geometry[2] is None
    with pytest.raises(AttributeError, match="delta"):  # the same slot without the declaration: strict, as every other role
        PreparedBwdLaunch(replace(spec, standalone_only_roles=()), binding)
    with pytest.raises(AttributeError, match="statz"):  # a misspelled role is still a plan-build failure, declaration or not
        PreparedBwdLaunch(replace(spec, attributes=("q", "statz", delta)), binding)
    assert BwdLaunchSpec(**base, native_binding=False).standalone_only_roles == (), "the default: every role is a graph attribute"


def test_prepared_sm107_bind_holds_the_two_appended_slots_independently():
    """``prepared.bind`` over the half row's eleven-slot operand tuple, slot by slot.  For each of the four specializations of
    the two appended slots (slot 9 ``seq_kv`` x slot 10 ``delta``, each None-specialized or an operand) and each of the four
    execute shapes (each buffer given or not), bind accepts exactly the matching shape -- the frame carries the two pointers in
    the slots' order -- and refuses the other three with a ValueError naming the slot: ``was not compiled into this
    specialization`` for a buffer the plan did not ask for, ``is required by this specialization`` for one it did (slot 9
    reported first when both are off).  ``bind`` only builds the frame, so nothing launches; the launch spec is hand-built with the
    nine tensor slots None-specialized, so this runs on any CUDA device -- the compiled plans' twin is the Rubin
    ``test_adapter_per_batch_kv_lengths_compose_with_the_gate_kernels_external_delta``."""
    from itertools import product

    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.prepared import BwdLaunchSpec, Operand, bind
    from cudnn.sdpa.fwd.prepared import facts_of_tensor

    b, hq, s_pad = 3, 2, 512
    roles, attributes = prepared_sm107.ROLES_F16, prepared_sm107.ATTRIBUTES_F16
    assert roles[9] == "seq_kv" and roles[10] == prepared_sm107.EXTERNAL_DELTA_ROLE and len(roles) == len(attributes) == 11
    seq_kv_op = Operand("int32", (b,), (1,), b, 4, 4)
    delta_op = Operand("float32", (b, hq, s_pad), (hq * s_pad, s_pad, 1), b * hq * s_pad, 16, 4)
    lens = torch.tensor([512, 300, 0], dtype=torch.int32, device="cuda")
    delta = torch.zeros(b, hq, s_pad, device="cuda")
    ws = torch.empty(16, device="cuda", dtype=torch.uint8)
    for with_lens, with_delta in product((False, True), repeat=2):
        operands = (None,) * 9 + (seq_kv_op if with_lens else None, delta_op if with_delta else None)
        spec = BwdLaunchSpec(
            None,
            None,
            operands,
            0,
            0,
            1.0,
            "probe",
            length_form=False,
            roles=roles,
            attributes=attributes,
            scale_log2=False,
            standalone_only_roles=(prepared_sm107.EXTERNAL_DELTA_ROLE,),
            native_binding=False,
        )
        for give_lens, give_delta in product((False, True), repeat=2):
            facts = {"seq_kv": facts_of_tensor(lens if give_lens else None), "delta": facts_of_tensor(delta if give_delta else None)}
            if (give_lens, give_delta) == (with_lens, with_delta):
                frame = bind(spec, facts, ws.data_ptr(), 0)
                assert frame[:9] == [None] * 9
                assert frame[9] == (lens.data_ptr() if with_lens else None) and frame[10] == (delta.data_ptr() if with_delta else None)
                continue
            slot, given = ("seq_kv", give_lens) if give_lens != with_lens else ("delta", give_delta)
            verb = "was not compiled into this specialization" if given else "is required by this specialization"
            with pytest.raises(ValueError, match=f"{slot} {verb}"):
                bind(spec, facts, ws.data_ptr(), 0)


def test_half_adapter_external_delta_execute_contract_fires_before_compile():
    """At execute, BEFORE ``compile()`` (no artifact, no launch -- so it runs on any CUDA host): both directions of the plan
    fact, then the exact ``dot_do_o`` layout -- fp32, contiguous ``(B, H_q, S_q_pad)`` (the PADDED extent, zeros past S_q),
    the plan's device, a 16-byte base -- each a ValueError naming ``delta_tensor``.  The Rubin half of the claim is
    ``test_external_delta_is_bitwise_the_chains_own_pre_pass``."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    own = _adapter(SdpaBwdDslSm107, b=1, hq=4, hkv=2, sq=500, skv=512, is_causal=True)
    ext = _adapter(SdpaBwdDslSm107, b=1, hq=4, hkv=2, sq=500, skv=512, is_causal=True, external_delta=True)
    dummy = torch.empty(1, device="cuda")  # never bound: every reject below fires before the bind
    args = {name + "_tensor": dummy for name in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")}
    args["workspace"] = dummy
    good = torch.zeros(1, 4, 512, device="cuda")
    with pytest.raises(ValueError, match="external_delta=False"):
        own.execute(**args, delta_tensor=good)
    with pytest.raises(ValueError, match="delta_tensor is required"):
        ext.execute(**args)
    with pytest.raises(ValueError, match="must be fp32"):
        ext.execute(**args, delta_tensor=good.to(torch.bfloat16))
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        ext.execute(**args, delta_tensor=torch.zeros(1, 4, 1024, device="cuda")[:, :, ::2])
    with pytest.raises(ValueError, match=r"CONTIGUOUS \[B, H_q, S_q_pad\] = \(1, 4, 512\)"):
        ext.execute(**args, delta_tensor=torch.zeros(1, 4, 500, device="cuda"))  # the REAL S_q: the chain's layout is the padded one
    with pytest.raises(ValueError, match="plan's device"):
        ext.execute(**args, delta_tensor=torch.zeros(1, 4, 512))
    with pytest.raises(ValueError, match="16-byte aligned"):
        ext.execute(**args, delta_tensor=torch.zeros(1 * 4 * 512 + 1, device="cuda")[1:].view(1, 4, 512))
    assert own._compiled is None and ext._compiled is None, "a reject must fire before compile()"


@requires_rubin
# S_q > S_kv is TOP-LEFT only: the frontend validator rejects a bottom-right causal graph with max_s_q > max_s_kv
# ("virtually slice the Q tensor") before any engine runs, so that combination is not a shape this row can be asked for.
@pytest.mark.parametrize(
    "sq,skv,bottom_right,left",
    [
        (512, 512, False, None),
        (512, 1024, True, None),
        (768, 512, False, None),
        (1024, 1024, False, 640),  # W = 639: the window's bounds land 63 rows into a k tile
        (1024, 1024, False, 200),  # W = 199: a window that is neither a k-tile nor a q-tile multiple
        (500, 1024, True, None),  # bottom-right with a ragged S_q: shift 524, the LO floor one q tile below the band
        (500, 1024, True, 200),  # ... plus a window: both edges off every tile grid
        (1024, 1024, None, 200),  # a window WITHOUT a causal diagonal (causal_diag=False)
    ],
    ids=["square", "br-rect-kv", "tl-rect-q", "swa640", "swa200", "br-ragged-sq", "br-ragged-sq-swa200", "swa200-no-causal"],
)
def test_stage3_causal_k_trim_is_bitwise_the_untrimmed_rendering(monkeypatch, sq, skv, bottom_right, left):
    """The dK / dQ GEMMs' two-sided K-trim (plan s5: LO on dK, HI on dQ, over the kv-major workspace; the window bounds
    the other side of each) is numerically INERT: rendering both GEMMs untrimmed (``STAGE3_CAUSAL_TRIM = False``, every
    k tile read over a zero-filled workspace) must give the SAME BITS for dQ / dK / dV.  A trim mode paired with the
    wrong operand major, a shift off by the bottom-right diagonal, or a window bound rounded INWARD drops real tiles
    here and shows up as a non-zero diff.  The trimmed run gets a POISONED workspace: it must not read a tile the kernel
    skipped (the poisoned-workspace test proves that per mask; here it makes the bitwise compare two proofs in one)."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    causal = bottom_right is not None
    kw = dict(use_causal_mask_bottom_right=True) if bottom_right else (dict(use_causal_mask=True) if causal else {})
    if left is not None:
        kw["diagonal_band_left_bound"] = left
    keep = _causal_keep(sq, skv, bottom_right=bool(bottom_right), left=left, causal=causal)
    trimmed = _run(b=2, hq=4, hkv=2, sq=sq, skv=skv, keep=keep, poison=float("nan"), ws_poison=0xFF, **kw).check()
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    untrimmed = _run(b=2, hq=4, hkv=2, sq=sq, skv=skv, keep=keep, poison=float("nan"), **kw).check()
    for name, x, y in zip(("dQ", "dK", "dV"), trimmed.outs[0], untrimmed.outs[0]):
        n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
        assert n_diff == 0, f"{name}: trimmed vs untrimmed stage 3 differ in {n_diff} elements (max|diff|={(x.float() - y.float()).abs().max().item():.3e})"


@requires_rubin
@pytest.mark.parametrize(
    "case",
    [
        dict(sq=512, skv=512),
        dict(sq=512, skv=512, causal=True),
        dict(sq=512, skv=1024, causal=True, bottom_right=True),
        dict(sq=500, skv=1024, causal=True, bottom_right=True),
        dict(hq=8, hkv=2, sq=512, skv=768),
        dict(hq=4, hkv=2, sq=256, skv=256, causal=True),
        dict(b=1, hq=2, sq=1024, skv=2048),
    ],
    ids=["dense", "causal", "bottom-right", "bottom-right-ragged-sq", "gqa4", "one-kv-block-causal", "8-kv-blocks"],
)
def test_stage3_d256_rendering_is_bitwise_the_padded_one(monkeypatch, case):
    """The d = 256 stage-3 rendering (``cgrp_tile_mn = (256, 256)``: cluster 2x1, one 256 x 256 tile per pair, the
    accumulator double-buffered -- PERF_PARITY_DESIGN section A) must give the SAME BITS for dQ / dK / dV as the SM100
    chain's (512, 512) rendering, which at d = 256 computes 256 columns of padding per cluster tile.  Bitwise is the
    right gate: both walk the same 64-wide k tiles in the same order with the same 256x256x16 instruction into an fp32
    accumulator, so every output element sees the identical reduction; a difference means the new row's tile / TMA
    box / multicast constants disagree with each other, not rounding.  ``STAGE3_D256_TILE = False`` is the twin."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    case = dict(case)
    causal, bottom_right = case.pop("causal", False), case.pop("bottom_right", False)
    kw = {}
    keep = None
    if causal:
        kw = dict(use_causal_mask_bottom_right=True) if bottom_right else dict(use_causal_mask=True)
        keep = _causal_keep(case["sq"], case["skv"], bottom_right=bottom_right)
    case.setdefault("b", 2)
    case.setdefault("hq", 4)
    case.setdefault("hkv", 2)
    assert sm107.STAGE3_D256_TILE, "the d256 rendering is what ships; the pin flips it OFF for the twin"
    d256 = _run(keep=keep, poison=float("nan"), **case, **kw).check()
    monkeypatch.setattr(sm107, "STAGE3_D256_TILE", False)
    padded = _run(keep=keep, poison=float("nan"), **case, **kw).check()
    for name, x, y in zip(("dQ", "dK", "dV"), d256.outs[0], padded.outs[0]):
        n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
        assert n_diff == 0, f"{name}: d256 vs padded stage-3 rendering differ in {n_diff} elements (max|diff|={(x.float() - y.float()).abs().max().item():.3e})"


@requires_rubin
@pytest.mark.parametrize("s", [1024, 2048])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("hq,hkv", [(8, 2), (32, 2)], ids=["gqa8-2", "gqa32-2"])
def test_stage3_single_launch_dq_is_bitwise_the_per_member_launches(monkeypatch, hq, hkv, causal, s):
    """Under GQA the shipped dQ GEMM is ONE launch per head chunk -- its rendering indexes B = K by ``h // group``
    (``MatmulTemplateParams.b_head_group = group``) over the whole dS and dQ chunk -- where it used to be one launch per group
    MEMBER over every ``group``-th Q head (16 under-one-wave launches at H_q / H_kv = 16).  Both pair every Q head with the same K
    head and walk the same k tiles per output tile into an fp32 accumulator, so dQ must be the SAME BITS -- and dK / dV, which the
    change never touches.  ``DQ_SINGLE_LAUNCH = False`` is the twin (``b_head_group = 1``, the per-member loop); both runs are
    also held to the fp64 oracle.  A difference here is a head pairing or a descriptor extent, not rounding."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(use_causal_mask=True) if causal else {}
    keep = _causal_keep(s, s) if causal else None
    assert sm107.DQ_SINGLE_LAUNCH, "one dQ launch per chunk is what ships; the pin flips it OFF for the twin"
    single = _run(b=1, hq=hq, hkv=hkv, sq=s, skv=s, keep=keep, poison=float("nan"), **kw).check()
    monkeypatch.setattr(sm107, "DQ_SINGLE_LAUNCH", False)
    members = _run(b=1, hq=hq, hkv=hkv, sq=s, skv=s, keep=keep, poison=float("nan"), **kw).check()
    for name, x, y in zip(("dQ", "dK", "dV"), single.outs[0], members.outs[0]):
        n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
        assert n_diff == 0, (
            f"{name}: the single dQ launch vs the per-member launches differ in {n_diff} of {x.numel()} elements "
            f"(max|diff|={(x.float() - y.float()).abs().max().item():.3e}, first at {(x.view(torch.int16) != y.view(torch.int16)).nonzero()[0].tolist()})"
        )


# --------------------------------------------------------------------------- the prepared launch (Rubin): the graph binds its pack into ONE artifact


def _prepared_case(dt=torch.bfloat16, causal=True, b=2, hq=4, hkv=2, sq=512, skv=512, seed=0):
    """Build + PIN + execute one half-row graph and hand back everything a prepared-launch pin needs: the graph, its
    port handles, the live tensors, the pack, the workspace and the fp64 oracle.  Inputs on a CPU generator (SM-count
    independent).  Checks the first execute before returning."""
    from types import SimpleNamespace

    group = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s_, h):
        return torch.randn(bb, s_, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do, k, v = draw(b, sq, hq), draw(b, sq, hq), draw(b, skv, hkv), draw(b, skv, hkv)
    keep = _causal_keep(sq, skv) if causal else None
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    kw = dict(use_causal_mask=True) if causal else {}
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dt=dt, scale="default", **kw)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8).fill_(0xBD)
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=lse.unsqueeze(-1).contiguous())
    tensors.update(
        dq=_bshd_empty(b, sq, hq, _D, dt, fill=float("nan")),
        dk=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
        dv=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
    )
    refs = dict(q=t["q"], k=t["k"], v=t["v"], o=t["o"], do=t["do"], stats=t["stats"], dq=dq_t, dk=dk_t, dv=dv_t)
    pack = {refs[name]: value for name, value in tensors.items()}
    g.execute(pack, ws)
    torch.cuda.synchronize()
    case = SimpleNamespace(
        graph=g,
        refs=refs,
        tensors=tensors,
        pack=pack,
        workspace=ws,
        keep=keep,
        group=group,
        dt=dt,
        expected=(dq_r, dk_r, dv_r),
        b=b,
        hq=hq,
        hkv=hkv,
        sq=sq,
        skv=skv,
        causal=causal,
    )
    assert g._compiled_plans[g._plan_index]._prepared is not None, "the sm107 row must lower through the prepared-launch contract"
    _check_prepared(case)
    return case


def _check_prepared(case, tensors=None, expected=None):
    tensors = case.tensors if tensors is None else tensors
    expected = case.expected if expected is None else expected
    for name, key, want in zip(("dQ", "dK", "dV"), ("dq", "dk", "dv"), expected):
        _check(name, tensors[key], want, case.dt)


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("causal", [False, True])
def test_prepared_sm107_rebind_stream_and_replay(dt, causal):
    """The plan's prepared launch rebinds fresh buffers, follows the HANDLE's stream (not the ambient one) and captures
    into a CUDA graph whose replay recomputes new inputs -- the contract every prepared backward shares (the sm100 pin)."""
    case = _prepared_case(dt=dt, causal=causal)
    tensors = {name: value.clone() for name, value in case.tensors.items()}
    pack = {case.refs[name]: value for name, value in tensors.items()}
    workspace = torch.empty_like(case.workspace).fill_(0xBD)
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()

    def refresh():
        tensors["q"].mul_(0.75)
        tensors["do"].mul_(1.25)
        o64, lse64, all_masked, dq, dk, dv = _reference64(tensors["q"], tensors["k"], tensors["v"], tensors["do"], case.keep, case.group)
        tensors["o"].copy_(o64.to(dt))
        lse = lse64.float()
        if all_masked is not None:
            lse = lse.masked_fill(all_masked, 0.0)
        tensors["stats"].copy_(lse.unsqueeze(-1))
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        workspace.fill_(0xBD)
        return dq, dk, dv

    try:
        expected = refresh()
        stream.wait_stream(torch.cuda.current_stream())
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            case.graph.execute(pack, workspace, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        _check_prepared(case, tensors, expected)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(other):
                case.graph.execute(pack, workspace, handle=handle)
        expected = refresh()
        capture.replay()
        torch.cuda.synchronize()
        _check_prepared(case, tensors, expected)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@requires_rubin
@pytest.mark.parametrize("hkv", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("sq,skv", [(512, 512), (500, 500)], ids=["aligned", "padded"])
def test_prepared_sm107_execute_has_no_tensor_wrapping(hkv, sq, skv, monkeypatch):
    """Execute rebuilds no tensor views, allocates nothing, compiles nothing and never synchronizes -- the padding copies,
    the seq_kv fill and the dS zero-fill are launches of the artifact, not torch ops (the pin that proves the torch
    path is gone; a padded shape exercises every staging kernel)."""
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver

    case = _prepared_case(causal=True, hkv=hkv, sq=sq, skv=skv)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared backward rebuilt tensor operands, allocated, synchronized or compiled")

    with monkeypatch.context() as patcher:
        for name in ("view", "reshape", "as_strided", "permute", "transpose", "copy_", "zero_", "fill_", "contiguous"):
            patcher.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full"):
            patcher.setattr(torch, name, forbidden)
        patcher.setattr(WorkspaceCarver, "__init__", forbidden)
        patcher.setattr(cute, "compile", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            case.graph.execute(case.pack, case.workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    _check_prepared(case)


@requires_rubin
@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "dq", "dk", "dv"])
def test_prepared_sm107_standalone_rejects_changed_layout(role):
    """The standalone adapter binds the SAME fixed artifact: an operand whose strides differ from the plan's is refused
    before any stage launches (a bare ValueError naming the geometry, never a wrong launch)."""
    from dataclasses import replace
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    case = _prepared_case()
    api = SdpaBwdDslSm107(**{"sample_" + name: value for name, value in case.tensors.items()}, is_causal=True, scale_softmax=_D**-0.5)
    api.check_support()
    api.compile()
    launches = []
    api._prepared = replace(api._prepared, fn=lambda *args: launches.append(args))
    args = {name + "_tensor": value for name, value in case.tensors.items()}
    args[role + "_tensor"] = case.tensors[role].contiguous()
    assert args[role + "_tensor"].stride() != case.tensors[role].stride()
    with pytest.raises(ValueError, match="runtime geometry"):
        api.execute(**args, workspace=case.workspace)
    assert not launches


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("sq", [512, 500], ids=["aligned", "padded"])
def test_external_delta_is_bitwise_the_chains_own_pre_pass(dt, sq):
    """A standalone plan built with ``external_delta=True`` and fed the delta the chain's OWN first launch wrote (read back
    out of a sibling plan's workspace region) returns dQ / dK / dV ``torch.equal`` the sibling's -- the same artifact minus
    the ``dot`` launch, reading the caller's tensor where the sibling reads its region.  Also pinned: the sibling's pad rows
    are the zeros the contract asks the caller for; the external plan launches exactly one kernel fewer (CUPTI, when
    available); a wrong-shape delta on the compiled plan is refused with no launch."""
    from dataclasses import replace

    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import R_DELTA

    b, hq, hkv, skv = 2, 4, 2, 512
    group = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(3)

    def draw(bb, s_, h):
        return torch.randn(bb, s_, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do, k, v = draw(b, sq, hq), draw(b, sq, hq), draw(b, skv, hkv), draw(b, skv, hkv)
    keep = _causal_keep(sq, skv)
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=lse.unsqueeze(-1).contiguous())
    samples = dict(tensors, dq=_bshd_empty(b, sq, hq, _D, dt), dk=_bshd_empty(b, skv, hkv, _D, dt), dv=_bshd_empty(b, skv, hkv, _D, dt))

    def build(external):
        api = SdpaBwdDslSm107(**{"sample_" + name: value for name, value in samples.items()}, is_causal=True, scale_softmax=_D**-0.5, external_delta=external)
        api.check_support()
        api.compile()
        return api

    def run(api, delta=None):
        grads = dict(
            dq=_bshd_empty(b, sq, hq, _D, dt, fill=float("nan")),
            dk=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
            dv=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
        )
        ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8).fill_(0xBD)
        api.execute(
            **{name + "_tensor": value for name, value in tensors.items()},
            **{name + "_tensor": value for name, value in grads.items()},
            workspace=ws,
            delta_tensor=delta,
        )
        torch.cuda.synchronize()
        return grads, ws

    own, ext = build(False), build(True)
    # slot 9 (the per-batch kv lengths) stays None-specialized on both plans; slot 10 is the delta, bound on the external plan only
    assert own._prepared.roles[9] == "seq_kv" and own._prepared.operands[9] is None and ext._prepared.operands[9] is None
    assert own._prepared.operands[10] is None and ext._prepared.operands[10] is not None and own._prepared.roles[10] == prepared_sm107.EXTERNAL_DELTA_ROLE
    grads_own, ws_own = run(own)
    for name, want in zip(("dq", "dk", "dv"), (dq_r, dk_r, dv_r)):
        _check(name, grads_own[name], want, dt)
    # the chain's own delta: region R_DELTA of the sibling's carve, [B, H_q, S_q_pad] fp32, zeros on the pad rows
    offset, shape, _strides = prepared_sm107._regions(own, prepared_sm107._REGION_SLOTS_F16)[0][R_DELTA]
    assert shape == own.external_delta_shape == ext.external_delta_shape == (b, hq, -(-sq // 128) * 128)
    delta = ws_own[offset : offset + 4 * math.prod(shape)].view(torch.float32).view(*shape).clone()
    assert torch.isfinite(delta).all() and torch.equal(delta[:, :, sq:], torch.zeros_like(delta[:, :, sq:]))
    grads_ext, _ws_ext = run(ext, delta)
    for name in ("dq", "dk", "dv"):
        assert torch.equal(grads_ext[name], grads_own[name]), f"{name}: the external-delta plan differs from the chain's own"
    # one launch fewer (the `dot` kernel), counted with CUPTI: only the profiler's own start may fail (-> None); a failure from
    # run() propagates and the count assertion sits outside any handler, so a restored dot launch FAILS the test
    counts = cuda_launch_counts(lambda: run(own, None), lambda: run(ext, delta))
    if counts is None:
        print("\nlaunch count unverified here (no CUDA profiler activity: CUPTI unavailable)")
    else:
        assert counts[1] == counts[0] - 1, counts
        print(f"\nlaunches: own {counts[0]}, external delta {counts[1]}")
    # the compiled plan refuses a wrong delta before any launch
    launches = []
    ext._prepared = replace(ext._prepared, fn=lambda *args: launches.append(args))
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        run(ext, delta[:, :, :sq].contiguous() if sq % 128 else delta.transpose(1, 2))
    with pytest.raises(ValueError, match="delta_tensor is required"):
        run(ext, None)
    assert not launches


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_adapter_per_batch_kv_lengths_compose_with_the_gate_kernels_external_delta(dt):
    """A plan built with BOTH appended plan facts -- ``seq_kv_lens_present=True`` (slot 9, the caller's per-batch kv lengths)
    and ``external_delta=True`` (slot 10, the caller's delta) -- fed the delta the gated block's sigmoid-gate backward kernel
    emits (``has_delta``: ``rowsum(dO * O)`` over the dO it stored, in ``dot_do_o``'s order) together with the lengths, returns
    dQ / dK / dV ``torch.equal`` the UNFUSED per-batch-lengths plan's (``seq_kv_lens_present=True`` alone, its own ``dot``
    launch) over the same dO and lengths; that run is itself held to the fp64 oracle composing the lengths and the causal band,
    dead kv rows exactly zero.  Lengths [512, 300, 0] on S_kv = 512 (a full, a ragged and a DEAD entry); S_q = 500 is ragged
    too (S_q_pad = 512), so the gate kernel's zeroed pad tail is what the contract asks for.  Also pinned: the gate kernel's
    delta is the chain's own bit for bit (read back from the unfused plan's ``R_DELTA`` region); the slots -- both bound on
    the combined plan, exactly one on each single-fact sibling; and the compiled plans' refusals, each a ValueError before
    any launch -- the combined plan refuses an execute missing either operand, each single-fact plan refuses the other's."""
    from dataclasses import replace

    from cudnn.gated_attention_block.kernels.sigmoid_gate_bwd import compile_sigmoid_gate_bwd, run_sigmoid_gate_bwd
    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import R_DELTA
    from cudnn.sdpa.fwd.api_dsl import ws_align

    b, hq, hkv, sq, skv = 3, 4, 2, 500, 512
    lens = [512, 300, 0]
    group = hq // hkv
    s_pad = -(-sq // 128) * 128
    gen = torch.Generator(device="cpu").manual_seed(5)

    def draw(bb, s_, h):
        return torch.randn(bb, s_, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, k, v = draw(b, sq, hq), draw(b, skv, hkv), draw(b, skv, hkv)
    keep = _per_batch_keep(sq, skv, lens, causal=True)
    # The forward's O (storage-rounded) is what the gate kernel reads, together with the gated output's upstream gradient.
    o64, lse64, all_masked, _dq, _dk, _dv = _reference64(q, k, v, torch.zeros_like(q), keep, group)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    # The gate kernel's token-major [T, H, D] operands ARE the BSHD storage of O / dO (T = B * S_q; the per-batch s = S_q).
    o_tok = o.permute(0, 2, 1, 3).reshape(b * sq, hq, _D)
    dog = torch.randn(b * sq, hq, _D, generator=gen).to(device="cuda", dtype=dt)
    gate = (torch.randn(b * sq, hq, _D, generator=gen) * 3.0).to(device="cuda", dtype=dt)
    do_tok, dg = torch.empty_like(dog), torch.empty_like(dog)
    delta_gate = torch.full((b, hq, s_pad), float("nan"), device="cuda", dtype=torch.float32)
    recipe = compile_sigmoid_gate_bwd(dtype=dt, h=hq, d=_D, has_og=False, has_seq_lens=False, has_delta=True)
    run_sigmoid_gate_bwd(recipe, dog, o_tok, gate, do_tok, dg, s=sq, stream=torch.cuda.current_stream().cuda_stream, delta=delta_gate)
    torch.cuda.synchronize()
    assert torch.isfinite(delta_gate).all() and torch.equal(delta_gate[:, :, sq:], torch.zeros_like(delta_gate[:, :, sq:]))
    do = do_tok.view(b, sq, hq, _D).permute(0, 2, 1, 3)
    _o, _lse, _am, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group)
    stats = lse64.float().masked_fill(all_masked, 0.0).unsqueeze(-1).contiguous()
    lens_t = torch.tensor(lens, dtype=torch.int32, device="cuda")
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=stats)

    def build(**flags):
        samples = {"sample_" + name: value for name, value in tensors.items()}
        samples.update(sample_dq=_bshd_empty(b, sq, hq, _D, dt), sample_dk=_bshd_empty(b, skv, hkv, _D, dt), sample_dv=_bshd_empty(b, skv, hkv, _D, dt))
        api = SdpaBwdDslSm107(**samples, is_causal=True, scale_softmax=_D**-0.5, **flags)
        api.check_support()
        api.compile()
        return api

    def run(api, **appended):
        grads = dict(
            dq=_bshd_empty(b, sq, hq, _D, dt, fill=float("nan")),
            dk=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
            dv=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
        )
        ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8).fill_(0xBD)
        api.execute(
            **{name + "_tensor": value for name, value in tensors.items()},
            **{name + "_tensor": value for name, value in grads.items()},
            workspace=ws,
            **appended,
        )
        torch.cuda.synchronize()
        return grads, ws

    lengths_only, both, delta_only = build(seq_kv_lens_present=True), build(seq_kv_lens_present=True, external_delta=True), build(external_delta=True)
    for api, bound in ((lengths_only, (True, False)), (both, (True, True)), (delta_only, (False, True))):
        spec = api._prepared
        assert spec.roles[9] == "seq_kv" and spec.roles[10] == prepared_sm107.EXTERNAL_DELTA_ROLE and len(spec.operands) == 11
        assert (spec.operands[9] is not None, spec.operands[10] is not None) == bound
    assert both.external_delta_shape == (b, hq, s_pad)
    assert lengths_only.scratch_workspace_bytes() - both.scratch_workspace_bytes() == ws_align(b * hq * s_pad * 4), "the carve lost exactly the delta region"
    # the unfused per-batch-lengths run: the oracle's, dead rows exact zeros
    grads_ref, ws_ref = run(lengths_only, seq_kv_lens=lens_t)
    for name, want in zip(("dq", "dk", "dv"), (dq_r, dk_r, dv_r)):
        _check(name, grads_ref[name], want, dt)
    _assert_dead_kv_rows_exactly_zero(types.SimpleNamespace(outs=[tuple(grads_ref[name] for name in ("dq", "dk", "dv"))]), lens)
    # the gate kernel's delta is the chain's own: region R_DELTA of the unfused plan's carve, [B, H_q, S_q_pad] fp32
    offset, shape, _strides = prepared_sm107._regions(lengths_only, prepared_sm107._REGION_SLOTS_F16)[0][R_DELTA]
    assert shape == (b, hq, s_pad)
    delta_chain = ws_ref[offset : offset + 4 * math.prod(shape)].view(torch.float32).view(*shape)
    assert torch.equal(
        delta_gate, delta_chain
    ), f"the gate kernel's delta differs from the chain's dot_do_o: max|diff|={(delta_gate - delta_chain).abs().max().item():.3e}"
    # the combined plan: the same artifact minus the dot launch, both appended operands bound
    grads_both, _ws = run(both, seq_kv_lens=lens_t, delta_tensor=delta_gate)
    for name in ("dq", "dk", "dv"):
        assert torch.equal(grads_both[name], grads_ref[name]), f"{name}: the combined plan differs from the unfused per-batch-lengths run"
    # the refusals on the compiled plans, each before any launch
    launches = []
    for api in (lengths_only, both, delta_only):
        api._prepared = replace(api._prepared, fn=lambda *args: launches.append(args))
    with pytest.raises(ValueError, match="exactly when"):
        run(both, delta_tensor=delta_gate)  # the lengths missing
    with pytest.raises(ValueError, match="delta_tensor is required"):
        run(both, seq_kv_lens=lens_t)  # the delta missing
    with pytest.raises(ValueError, match="external_delta=False"):
        run(lengths_only, seq_kv_lens=lens_t, delta_tensor=delta_gate)  # a delta the plan did not ask for
    with pytest.raises(ValueError, match="exactly when"):
        run(delta_only, seq_kv_lens=lens_t, delta_tensor=delta_gate)  # lengths the plan did not ask for
    assert not launches


@requires_rubin
@pytest.mark.parametrize("role", ["q", "stats", "dq"])
@pytest.mark.parametrize("ordered", [False, True])
def test_prepared_sm107_raw_storage_and_explicit_overrides(role, ordered, monkeypatch):
    """Graph bindings are raw storage under the declared layout (a strided producer view binds); an explicit shape
    override that matches passes, one that changes the operation is refused before launch."""
    from dataclasses import replace

    case = _prepared_case()
    tensor = case.tensors[role]
    backing = torch.empty(tensor.numel() + 2, dtype=tensor.dtype, device="cuda")
    declared = backing.as_strided(tensor.shape, tensor.stride()).copy_(tensor)
    case.pack[case.refs[role]] = backing[::2]
    case.tensors[role] = declared
    ref = case.refs[role]
    kwargs = {}
    pack = case.pack
    if ordered:
        items = list(reversed(list(pack.items())))
        kwargs["tensor_uids"] = [t.get_uid() for t, _ in items]
        pack = [buffer for _, buffer in items]
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    case.graph.execute(pack, case.workspace, **kwargs)
    torch.cuda.synchronize()
    _check_prepared(case)
    kwargs.update(override_uids=[ref.get_uid()], override_shapes=[list(ref.get_dim())], override_strides=[list(ref.get_stride())])
    case.graph.execute(pack, case.workspace, **kwargs)
    torch.cuda.synchronize()
    _check_prepared(case)
    plan = case.graph._compiled_plans[case.graph._plan_index]
    launches = []
    monkeypatch.setattr(plan._prepared, "spec", replace(plan._prepared.spec, fn=lambda *args: launches.append(args)))
    kwargs["override_shapes"][0][2] //= 2
    with pytest.raises(ValueError, match="runtime geometry"):
        case.graph.execute(pack, case.workspace, **kwargs)
    assert not launches


@requires_rubin
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_prepared_backward_artifact_reloads_in_fresh_process(dtype, tmp_path):
    """The artifact exports to the compiled-plan cache and a second process runs the plan from it without a JIT."""
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm107", "dense", dtype, tmp_path)


def test_sm107_adapter_has_no_torch_execute_path():
    """The adapter's execute is ``facts -> bind -> the artifact``: no torch view / copy / fill, no workspace carver, no
    JIT at execute.  A static pin, so the torch path cannot creep back one helper at a time."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    code = _code_only(Path(sm107.__file__).read_text())
    for needle in (
        "WorkspaceCarver",
        "_torch_stream_context",
        "cute.compile(",
        "from_dlpack",
        ".copy_(",
        ".zero_(",
        ".fill_(",
        ".view(",
        ".permute(",
        "matmul_bh",
    ):
        assert needle not in code, f"api_dsl_sm107: {needle!r} is a torch-path spelling"
    assert "execute_standalone(" in code and "self._prepared = " in code


# --------------------------------------------------------------------------- the 2x2-datapath twin (api_dsl_sm107.BWD_D256_2X2)


_TWIN_PROFILES = [pytest.param(1, id="p1"), pytest.param(2, id="p2")]  # config_d256_2x2.PROFILE_SM100 / PROFILE_SM107_INTERLEAVED


def _twin_on(monkeypatch, profile):
    """Flip the twin on at ``profile`` for one test: both constants are read at CALL time by the half adapter."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    monkeypatch.setattr(sm107, "BWD_D256_2X2", True)
    monkeypatch.setattr(sm107, "BWD_D256_2X2_PROFILE", profile)


class _LoadSpy:
    """Records every (kernel file, datapath_2x2_profile) the half adapter loads while a plan builds.  The pin that the body
    which RAN is the one the twin selected: the prepared host is compiled from exactly the module ``load_template`` hands
    back, and the two bodies' kernel NAMES are identical under the profiler (``cudnn_kernel__kernel_...``), so a name
    match cannot tell them apart."""

    def __init__(self, monkeypatch):
        import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

        self.loads = []
        real = sm107.load_template

        def spy(path, params, tag="template"):
            self.loads.append((os.path.basename(path), getattr(params, "datapath_2x2_profile", None), tag))
            return real(path, params, tag=tag)

        monkeypatch.setattr(sm107, "load_template", spy)

    def main_kernels(self):
        return sorted({(f, p) for f, p, _t in self.loads if f.startswith("bprop_d256")})

    def assert_body(self, profile):
        """Exactly one main-kernel body was loaded: the 2x2 file at ``profile`` (1 / 2), or the 4x1 file at 0."""
        want = ("bprop_d256_2x2_f16.py", profile) if profile else ("bprop_d256_f16.py", 0)
        assert self.main_kernels() == [want], f"the plan loaded {self.main_kernels()}, expected {[want]} (loads: {self.loads})"


def test_2x2_twin_is_off_by_default_and_selects_the_2x2_body_when_on(monkeypatch):
    """``BWD_D256_2X2`` is read at CALL time: off (the shipped default) the half adapter loads the 4x1 body at profile 0 (its
    rendering byte-identical: the record only gains the appended field at its default); on, the shared 2x2 body at
    ``BWD_D256_2X2_PROFILE`` (default 2, the Rubin interleaved twin; 1 = the SM100 row's body) with its own tag, the profile
    copied INTO the TemplateParams record (it reaches the template hash), the stage-3 granularity still the 256-row pair.
    An illegal profile raises; the fp8 row ignores the constants."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107
    from cudnn.sdpa.bwd import config_d256_2x2 as c2
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107, SdpaBwdDslSm107Fp8

    assert sm107.BWD_D256_2X2 is False, "the 4x1 body ships; the twin flips on only at <= 1.00x on Rubin"
    assert sm107.BWD_D256_2X2_PROFILE == c2.PROFILE_SM107_INTERLEAVED == 2, "the twin's profile is the design's interleaved one"
    api = _adapter(SdpaBwdDslSm107)
    assert (api._kernel_file(), api._template_tag(), api._datapath_2x2_profile()) == ("sm107/bprop_d256_f16.py", "sdpa_bwd_sm107_main_f16", 0)
    assert api._template_params() == _cfg_records_equal_default(api)
    monkeypatch.setattr(sm107, "BWD_D256_2X2", True)
    assert (api._kernel_file(), api._template_tag(), api._datapath_2x2_profile()) == (
        "bprop_d256_2x2_f16.py",
        "sdpa_bwd_sm107_main_2x2",
        c2.PROFILE_SM107_INTERLEAVED,
    )
    assert api._template_params().datapath_2x2_profile == 2
    mod = _load_2x2(2)
    assert api._stage3_gran(mod) == c2.kv_pad_rows_2x2(mod.CFG) == 256 and mod.DESC_VERSION == 1 and mod._KV_BLOCK_ROWS == 256
    monkeypatch.setattr(sm107, "BWD_D256_2X2_PROFILE", c2.PROFILE_SM100)
    assert (api._kernel_file(), api._datapath_2x2_profile(), api._template_params().datapath_2x2_profile) == ("bprop_d256_2x2_f16.py", 1, 1)
    mod1 = _load_2x2(1)
    assert api._stage3_gran(mod1) == 256 and mod1.DESC_VERSION == 0 and mod1._KV_BLOCK_ROWS == 128, "profile 1: 128-row kv block, 256-row write pair"
    for bad in (0, 3, None):
        monkeypatch.setattr(sm107, "BWD_D256_2X2_PROFILE", bad)
        with pytest.raises(ValueError, match="BWD_D256_2X2_PROFILE"):
            api._datapath_2x2_profile()
    monkeypatch.setattr(sm107, "BWD_D256_2X2_PROFILE", c2.PROFILE_SM107_INTERLEAVED)
    fp8 = _adapter(SdpaBwdDslSm107Fp8, dt=torch.float8_e4m3fn, grad_dt=torch.float8_e4m3fn)
    assert (fp8._kernel_file(), fp8._datapath_2x2_profile()) == ("sm107/bprop_d256_fp8.py", 0), "the twin is f16-only"


def _cfg_records_equal_default(api):
    """The half adapter's record with the appended field at its default -- what a pre-field adapter built."""
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams

    p = api._template_params()
    return TemplateParams(**{k: v for k, v in p.__dict__.items() if k != "datapath_2x2_profile"}, datapath_2x2_profile=0)


def _load_2x2(profile):
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams

    return load_template(
        _sm100_kernel_path("bprop_d256_2x2_f16.py"), TemplateParams(dtype_qkv=DTYPE_BF16, datapath_2x2_profile=profile), tag=f"sm107_twin_p{profile}"
    )


def test_2x2_profile_2_is_the_rubin_interleaved_layout():
    """Profile 2's facts the Rubin board run exercises: two 64-row sub-blocks per CTA (256-row block), 322 KiB of slabs, the
    sP root at 256 KiB -> descriptor version 1, one warpgroup per sub-block (64 q cols per lane), L_CNT 256, the lookahead
    MMA order, and the 4x1 body's 224 / 56 register split (0 / 0 spills at sm_107a; 91 / 129 at profile 1's 176 / 152)."""
    from cudnn.sdpa.bwd import config_d256_2x2 as c2

    mod = _load_2x2(2)
    cfg = mod.CFG
    assert (cfg.KV_SUBBLOCKS, cfg.ROWS_PER_CTA, cfg.KV_BLOCK_ROWS, cfg.COLS_PER_LANE, cfg.L_CNT, mod.DESC_VERSION) == (2, 128, 256, 64, 256, 1)
    assert c2.smem_bytes_2x2(cfg) == 329728 and dict(c2.desc_roots_2x2(cfg))["sP[0][0]"] == 262144
    assert c2.tmem_layout_2x2(cfg).USED_COLS == 512
    assert (cfg.SOFTMAX_REGS, cfg.OTHER_REGS, mod.MMA_LOOKAHEAD) == (224, 56, True)


# The twin's GPU matrix on Rubin, both profiles: every mask arm the row serves, the GQA ratios the perf shape and the dQ
# single-launch pin use (8/2, 32/2), non-tile seqlens (incl. the padded-kv arm at 257 x 129), one kv write pair with many q
# tiles and many pairs with one q tile, fp16.  ``causal`` / ``bottom_right`` / ``left`` are the band spellings of
# ``_MASK_POISON_CASES``; the rest are ``_run`` kwargs.
_TWIN_CASES = {
    "dense": dict(),
    "fp16-dense": dict(dt=torch.float16),
    "causal": dict(causal=True),
    "bottom-right": dict(sq=512, skv=1024, causal=True, bottom_right=True),
    "bottom-right-ragged-sq": dict(b=1, sq=500, skv=1024, causal=True, bottom_right=True),
    "swa640": dict(sq=1024, skv=1024, causal=True, left=640),
    "gqa8-2": dict(hq=8, hkv=2, sq=256, skv=256),
    "gqa32-2-causal": dict(hq=32, hkv=2, sq=256, skv=512, causal=True),
    "768x1280": dict(sq=768, skv=1280),
    "500x500-causal": dict(sq=500, skv=500, causal=True),
    "257x129": dict(sq=257, skv=129),
    "one-kv-pair-8-q-tiles": dict(hq=2, sq=1024, skv=256),
    "8-kv-pairs-1-q-tile": dict(hq=2, sq=128, skv=2048),
}


def _twin_case_kw(case):
    case = dict(case)
    causal, bottom_right, left = case.pop("causal", False), case.pop("bottom_right", False), case.pop("left", None)
    kw = dict(use_causal_mask_bottom_right=True) if bottom_right else (dict(use_causal_mask=True) if causal else {})
    if left is not None:
        kw["diagonal_band_left_bound"] = left
    keep = _causal_keep(case.get("sq", 512), case.get("skv", 512), bottom_right=bottom_right, left=left, causal=bool(causal)) if (causal or left) else None
    return case, kw, keep


@requires_rubin
@pytest.mark.parametrize("profile", _TWIN_PROFILES)
@pytest.mark.parametrize("case", list(_TWIN_CASES), ids=list(_TWIN_CASES))
def test_twox2_twin_accepts_on_rubin(monkeypatch, profile, case):
    """The 2x2 twin on the Rubin board against the fp64 oracle (the engine pinned, the constants flipped), profile 1 (the SM100
    row's body, descriptor version 0) and profile 2 (interleaved, descriptor version 1); the load spy pins that the 2x2 body
    at that profile is the one the plan was built from.  First board run 2026-10-01 (profile 2: all PASS, bitwise the 4x1)."""
    _twin_on(monkeypatch, profile)
    spy = _LoadSpy(monkeypatch)
    shape, kw, keep = _twin_case_kw(_TWIN_CASES[case])
    _run(keep=keep, poison=float("nan"), **shape, **kw).check()
    spy.assert_body(profile)


@requires_rubin
@pytest.mark.parametrize("profile", _TWIN_PROFILES)
@pytest.mark.parametrize("case", list(_MASK_POISON_CASES), ids=list(_MASK_POISON_CASES))
def test_twox2_twin_masked_stage3_reads_only_what_stage2_wrote(monkeypatch, profile, case):
    """``test_masked_stage3_reads_only_what_stage2_wrote`` over the twin: the kv WRITE-PAIR invariant (a 128-row block on
    profile 1 derives its q range from its 256-row pair; profile 2's 256-row block IS the pair) keeps the stage-3 K-trim
    reading only written dS tiles -- on a 0xFF-poisoned workspace, without the zero-fill."""
    _twin_on(monkeypatch, profile)
    shape, kw, keep = _mask_case_kw(_MASK_POISON_CASES[case])
    _run(keep=keep, poison=float("nan"), ws_poison=0xFF, **shape, **kw).check()


@requires_rubin
@pytest.mark.parametrize("profile", _TWIN_PROFILES)
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_twox2_twin_two_launches_are_bitwise_and_race_free(monkeypatch, profile, dt):
    """The two-launch race + determinism probe over the twin (``Producer.LEADER_RELEASE`` on the lane-written P ring, the
    explicit S / dP / P slot barriers), raw bits, causal GQA with several tiles per CTA."""
    _twin_on(monkeypatch, profile)
    run = _run(b=2, hq=4, hkv=2, sq=512, skv=768, dt=dt, keep=_causal_keep(512, 768), use_causal_mask=True, runs=3, poison=float("nan")).check()
    for which, (a, b_) in (("launch 2 vs 1 (race)", (run.outs[1], run.outs[0])), ("launch 3 vs 2 (determinism)", (run.outs[2], run.outs[1]))):
        for name, x, y in zip(("dQ", "dK", "dV"), a, b_):
            n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
            assert n_diff == 0, f"{name} {which}: {n_diff} elements differ, max|diff|={(x.float() - y.float()).abs().max().item():.3e}"


@requires_rubin
@pytest.mark.parametrize("profile", _TWIN_PROFILES)
@pytest.mark.parametrize(
    "shape",
    [dict(b=2, hq=4, hkv=2, sq=512, skv=768, causal=True), dict(b=1, hq=8, hkv=2, sq=1024, skv=1024), dict(b=1, hq=32, hkv=2, sq=512, skv=512, causal=True)],
    ids=["causal-gqa4-2", "dense-gqa8-2", "causal-gqa32-2"],
)
def test_twox2_twin_is_bitwise_the_4x1_body_on_rubin(monkeypatch, profile, shape):
    """The twin's dQ / dK / dV are BITWISE the shipped 4x1 body's (int16 views).  MEASURED 2026-10-01 on the board: 0 elements
    differ on profile 2 at causal GQA 4/2 512x768 -- the M = 128 cta_group::2 instruction's per-element K = 16 reduction
    tree matches the M = 256 one's, P / dS are the same per-element arithmetic, and the stage-3 GEMMs are shared.  Design
    open question 2 is thereby closed on the bitwise side; the documented fallback (both within the fp64-oracle tolerance)
    would re-open it, so a non-zero count FAILS here with the magnitude rather than silently tolerating it."""
    shape = dict(shape)
    causal = shape.pop("causal", False)
    kw = dict(keep=_causal_keep(shape["sq"], shape["skv"]), use_causal_mask=True) if causal else {}
    base_spy = _LoadSpy(monkeypatch)
    base = _run(**shape, **kw).check()
    base_spy.assert_body(0)
    _twin_on(monkeypatch, profile)
    twin_spy = _LoadSpy(monkeypatch)
    twin = _run(**shape, **kw).check()
    twin_spy.assert_body(profile)
    for name, x, y in zip(("dQ", "dK", "dV"), twin.outs[0], base.outs[0]):
        n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
        print(f"2x2 twin p{profile} vs 4x1 {name}: {n_diff} elements differ (max|diff| {(x.float() - y.float()).abs().max().item():.3e})")
        assert (
            n_diff == 0
        ), f"{name}: the 2x2 twin (profile {profile}) is not bitwise the 4x1 body: {n_diff} elements differ, max|diff| {(x.float() - y.float()).abs().max().item():.3e}"


# --------------------------------------------------------------------------- bitwise vs the pre-port kernel (Rubin; dumps under frost_dev/results)


_REF_PROBE_STEM = "bf16_b1h8s1024_dense"  # every reference-dump directory carries this dump


def _ref_dump_dir():
    """The reference-dump directory: ``FROST_BWD_D256_REF_DIR``, else the subdirectory of
    ``frost_dev/results/bwd_d256_sm107/`` that holds the probe dump -- of this checkout, or of the main checkout when
    this file lives in a ``.worktrees/<slug>`` worktree (frost_dev is untracked).  Named by content, not by the pre-port
    project."""
    if os.environ.get("FROST_BWD_D256_REF_DIR"):
        return Path(os.environ["FROST_BWD_D256_REF_DIR"])
    root = Path(__file__).resolve().parents[4]
    roots = [root] + ([root.parents[1]] if root.parent.name == ".worktrees" else [])
    for r in roots:
        hits = sorted((r / "frost_dev" / "results" / "bwd_d256_sm107").glob(f"*/{_REF_PROBE_STEM}.pt"))
        if hits:
            return hits[0].parent
    return roots[0] / "frost_dev" / "results" / "bwd_d256_sm107" / "reference_dumps"


def _load_ref_dump(stem):
    path = _ref_dump_dir() / f"{stem}.pt"
    if not path.is_file():
        pytest.skip(f"no reference dump ({path})")
    return torch.load(path, map_location="cpu", weights_only=False)


def _ref_bshd(t):
    """A dumped BSHD [B, S, H, D] tensor as the [B, H, S, D] view over BSHD memory the graph wants, on the device."""
    return t.to("cuda").permute(0, 2, 1, 3)


_F16_REF_STEMS = ["bf16_b1h8s1024_dense", "bf16_b1h8kv2s2048_causal", "bf16_b2h4s768x1280_dense"]


@requires_rubin
@pytest.mark.parametrize("stem", _F16_REF_STEMS)
def test_dv_is_bitwise_the_reference_kernel_and_dq_dk_within_storage_tolerance(stem):
    """Feed the port the dump's storage-rounded q / k / v / dO and the dump's ``lse`` (stats), and compare dV BITWISE
    against the pre-port kernel's per-Q-head ``dv_kern`` (MHA dumps) or its folded ``dv_red`` within one bf16 rounding
    (the GQA dump: the fold order may differ).  dV depends on P (from lse) and dO only -- not on delta, which the graph
    path recomputes -- so it is the one output the graph path can hold bitwise; any non-zero difference must be
    EXPLAINED (the f16 idesc ``k_dim`` -- plan Q6, the exp2 spelling, FMA contraction) before the fp64 oracle is trusted
    alone.  dQ / dK go through the ported GEMMs (a different accumulation order than the pre-port matmul) and are held
    to the storage tolerance against the dump's fp64 oracle AND the pre-port chain's own real-unit outputs."""
    ref = _load_ref_dump(stem)
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    assert m["storage_dtype"] == "torch.bfloat16" and int(m["mask_flags"]) in (0, 2) and not int(m["swa_window"]) and not int(m["padded"])
    dt = torch.bfloat16
    q, k, v, do = (_ref_bshd(ref[n]) for n in ("q", "k", "v", "do"))
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(ref["o"].to("cuda").to(dt))
    stats = ref["lse"].to("cuda").unsqueeze(-1).contiguous()
    kw = dict(use_causal_mask=True) if bool(m["causal"]) else {}
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dt=dt, scale=float(m["attn_scale_in"]), **kw)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    dq, dk, dv = _bshd_empty(b, sq, hq, _D, dt, float("nan")), _bshd_empty(b, skv, hkv, _D, dt, float("nan")), _bshd_empty(b, skv, hkv, _D, dt, float("nan"))
    g.execute({t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: stats, dq_t: dq, dk_t: dk, dv_t: dv}, ws)
    torch.cuda.synchronize()
    got_dv = dv.cpu().contiguous()
    if hq == hkv:
        want = ref["dv_kern"].permute(0, 2, 1, 3).contiguous()
        n_diff = (got_dv.view(torch.int16) != want.view(torch.int16)).sum().item()
        diff = (got_dv.float() - want.float()).abs()
        assert n_diff == 0, (
            f"dV is NOT bitwise the pre-port kernel: {n_diff} of {want.numel()} elements differ, max|diff|={diff.max().item():.3e} "
            f"({(diff.max() / (want.float().abs().max() * _BF16_ULP_REL)).item():.1f} bf16 ulps at |dV|max) -- explain before trusting the oracle alone (plan Q6)"
        )
    else:
        want = ref["dv_red"].permute(0, 2, 1, 3).contiguous()
        torch.testing.assert_close(
            got_dv.float(), want.float(), rtol=_BF16_ULP_REL, atol=1e-6, msg=lambda s: f"folded dV vs the pre-port head-reduce (one bf16 rounding allowed): {s}"
        )
    for name, got, oracle, chain in (("dQ", dq, ref["ref_dq"], ref["dq"]), ("dK", dk, ref["ref_dk"], ref["dk"])):
        got = got.cpu().float()
        torch.testing.assert_close(got, oracle.float(), **_TOL[dt], msg=lambda s, n=name: f"{n} vs the dump's fp64 oracle: {s}")
        torch.testing.assert_close(got, chain.float(), **_TOL[dt], msg=lambda s, n=name: f"{n} vs the pre-port chain's real-unit output: {s}")


@requires_rubin
@pytest.mark.parametrize("stem", _F16_REF_STEMS)
def test_dv_and_ds_workspace_are_bitwise_the_reference_kernel(stem):
    """KERNEL-level, through the body's documented ``## Launch ABI``::

        fn = compile(b, qh, kh, sq, skv)
        fn(q, do, k, v, dv, ds, lse, do_dot, seq_kv_lens, problem_size, attn_scale, head_base, batch_base, stream)

    with the dump's ``lse`` AND ``delta`` INJECTED (the graph path recomputes delta from a bf16 O with dot_do_o, so the
    dS workspace is comparable only here).  Same storage-rounded operands, same math (natural MMA order, P written into
    S_acc): ``dv`` (per-Q-head, [B, S_kv, H_q, D]) and the kv-major ``ds`` workspace ([B, H_q, S_kv, S_q], softmax scale
    included) must be BITWISE the pre-port kernel's ``dv_kern`` / ``ds_ws``.  Any non-zero difference must be EXPLAINED
    (the f16 idesc ``k_dim`` -- plan Q6, the exp2 spelling, FMA contraction) before the fp64 oracle is trusted alone.
    The workspace is zero-initialised like the pre-port driver's (a causal launch leaves the q tiles a kv block skips
    untouched); dv is NaN-poisoned (every kv block stores it, the forced fully-masked tile as zeros)."""
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.tile_dsl.constants import DTYPE_BF16

    ref = _load_ref_dump(stem)
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    assert m["storage_dtype"] == "torch.bfloat16" and int(m["mask_flags"]) in (0, 2) and sq % 128 == 0 and skv % 256 == 0
    mod = _load_kernel("f16", dtype_qkv=DTYPE_BF16, window_right=0 if bool(m["causal"]) else None)
    fn = mod.compile(b, hq, hkv, sq, skv)
    dev = "cuda"
    q, do, k, v = (ref[n].to(dev).contiguous() for n in ("q", "do", "k", "v"))  # BSHD storage
    dv = torch.full((b, skv, hq, _D), float("nan"), device=dev, dtype=torch.bfloat16)
    ds = torch.zeros((b, hq, skv, sq), device=dev, dtype=torch.bfloat16)
    lse, delta = ref["lse"].to(dev).contiguous(), ref["delta"].to(dev).contiguous()
    seq_kv_lens = torch.full((b,), skv, dtype=torch.int32, device=dev)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(q, do, k, v, dv, ds, lse, delta, seq_kv_lens, (b, hq, hkv, sq, skv, hq, b, sq, skv), float(m["attn_scale_in"]), 0, 0, stream)
    torch.cuda.synchronize()
    for name, got, want in (("dv_kern", dv, ref["dv_kern"]), ("ds_ws", ds, ref["ds_ws"])):
        got, want = got.cpu().contiguous(), want.contiguous()
        assert got.shape == want.shape and got.dtype == want.dtype, (name, got.shape, got.dtype, want.shape, want.dtype)
        assert torch.isfinite(got.float()).all(), f"{name}: non-finite output (poison survived or NaN residue)"
        n_diff = (got.view(torch.int16) != want.view(torch.int16)).sum().item()
        diff = (got.float() - want.float()).abs()
        assert n_diff == 0, (
            f"{name} is NOT bitwise the pre-port kernel: {n_diff} of {want.numel()} elements differ, max|diff|={diff.max().item():.3e} "
            f"({(diff.max() / (want.float().abs().max() * _BF16_ULP_REL)).item():.1f} bf16 ulps at |{name}|max) -- explain before trusting the oracle alone (plan Q6)"
        )


# =========================================================================== static pins (host), both kernel families -- cloned from the forward Rubin suite


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_descriptor_version_matches_the_smem_budget(family):
    """A version-0 tcgen05 descriptor's ``start_address`` is 14 bits = a 256 KiB window; Rubin raises the per-CTA cap to
    327 KiB, and both bodies sit past 256 KiB (f16 322 KiB, fp8 306 KiB).  Every MMA / UTCCP ROOT of both layouts is
    below the line (config_sm107.desc_roots), so both are version 0 -- asserted against the module's own constant AND
    the config's live derivation, never a substring a comment could supply."""
    from cudnn.sdpa.bwd import config_sm107 as cfgmod

    mod = _load_kernel(family)
    assert mod.DESC_VERSION == cfgmod.desc_version(
        mod.CFG
    ), f"{mod.__name__}: DESC_VERSION={mod.DESC_VERSION} but the layout tally says {cfgmod.desc_version(mod.CFG)}"
    assert mod.DESC_VERSION == 0, f"{mod.__name__}: a root crossed 256 KiB -- move it or flip the derivation, do not delete the check"


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_every_smem_tile_takes_the_module_desc_version(family):
    """DESC_VERSION is only meaningful if EVERY ``SmemTile`` is built with it (the d512 MXFP8 forward shipped NaN on 100 %
    of cells from one re-literalled call site)."""
    code = _code_lines(_kernel_source(family))
    n_tiles = len(re.findall(r"\bSmemTile\(", code))
    assert n_tiles > 0, f"{family}: no SmemTile( in the kernel"
    n_wired = code.count("desc_version=DESC_VERSION")
    assert n_wired == n_tiles, f"{family}: {n_tiles} SmemTile(s) but {n_wired} wired to DESC_VERSION"
    assert not re.search(r"desc_version=[01]\b", code), f"{family}: a re-literalled desc_version bypasses DESC_VERSION"
    assert len(re.findall(r"^DESC_VERSION\b.*=", code, re.M)) == 1, f"{family}: exactly one DESC_VERSION definition"


# Ring-wait retry form: ONE module constant per kernel (``tile_dsl.barrier.wait(spin=...)``; the sign is PER KERNEL and
# measured).  WHICH sites are ring sites is the body's own classification, documented at its ``SPIN_RING_WAITS``
# definition (the fp8 body spins its per-q-iteration rings only; the f16 body also its per-kv-block K / V / dV
# handshakes) -- the pin records that classification as the (ring, idle) site COUNTS so a re-classified, re-literalled or
# newly added site cannot drift in silently, and forbids the spin on the two waits every body parks in for a whole tile:
# the scheduler payload and ``mb_tmem_dealloc``.  None = not yet pinned (the body is still moving): structural check only.
_RING_WAIT_SITES = {
    "f16": (23, 15),
    "fp8": (17, 24),
    "mxfp8": (20, 25),
}  # family -> (ring sites, idle sites); fp8 measured on 5e99bb9b, f16 after the P8 / drain fixes; mxfp8 = the fp8 body's sites + the mb_p_sf_consumed ring wait (before the P store) and its end-of-kernel drain,
# + 2 SOURCE sites: the MMA warp's loop Q.K block (its s_acc_empty + q_full waits) is spelled once per S issue order under
# cutlass.const_expr(S_LOOKAHEAD) -- one arm traces, so a binary still has 18 ring waits (test_sdpa_bwd_mxfp8_sm107 pins the per-arm count).
_IDLE_WAIT_TARGETS = ("mb_tmem_dealloc",)


def _wait_sites(code):
    """Every mbarrier wait call site: (target, balanced argument text).  Line-anchored (``<bars.>mb_X[...].wait(`` or the
    bare ``wait(sched.mb_...`` payload wait), paren-matched because black wraps some calls over several lines."""
    out = []
    for mt in re.finditer(r"^\s*(?:(?:bars\.)?(mb_\w+)(?:\[[^\]]*\])*\.wait\(|wait\((sched\.mb_\w+))", code, re.M):
        target = mt.group(1) or mt.group(2)
        i, depth = mt.end(), 1
        while depth:
            depth += (code[i] == "(") - (code[i] == ")")
            i += 1
        out.append((target, code[mt.end() : i - 1]))
    return out


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_ring_waits_take_the_module_spin_constant(family):
    mod = _load_kernel(family)
    assert isinstance(mod.SPIN_RING_WAITS, bool), f"{mod.__name__}: SPIN_RING_WAITS must be a bool module constant"
    code = _code_lines(_kernel_source(family))
    assert len(re.findall(r"^SPIN_RING_WAITS: bool = (?:True|False)$", code, re.M)) == 1, f"{family}: exactly one SPIN_RING_WAITS definition"
    assert not re.search(r"spin=(?:True|False)\b", code), f"{family}: a spin= literal at a call site bypasses SPIN_RING_WAITS"
    sites = _wait_sites(code)
    spun = [t for t, args in sites if "spin=SPIN_RING_WAITS" in args]
    assert sites, f"{family}: no mbarrier wait sites found"
    assert spun, f"{family}: no ring wait passes spin=SPIN_RING_WAITS (plan s4.12)"
    leaked = [t for t in spun if t.startswith(_IDLE_WAIT_TARGETS) or t.startswith("sched.")]
    assert not leaked, f"{family}: whole-tile idle waits must keep the sleeping form: {leaked}"
    assert code.count("spin=") == len(spun), f"{family}: a spin= outside a ring .wait( call"
    pinned = _RING_WAIT_SITES[family]
    if pinned is not None:
        n_ring, n_idle = pinned
        assert len(spun) == n_ring, f"{family}: {len(spun)} ring waits pass spin=SPIN_RING_WAITS, the classification says {n_ring}"
        assert len(sites) == n_ring + n_idle, f"{family}: {len(sites)} wait sites, expected {n_ring} ring + {n_idle} idle"


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_no_cluster_scope_release_arrive(family):
    """A ``MemScope.CLUSTER`` arrive without ``relaxed=True`` is a GPU-scope drain (MEMBAR.ALL.GPU + CGAERRBAR) per arrive
    site -- 47 -> 92 % of SOL on the pre-port cga2 kernels came from this flag alone.  The SASS twin below counts
    CGAERRBAR."""
    code = _code_lines(_kernel_source(family))
    bad = [ln for ln in code.splitlines() if "MemScope.CLUSTER" in ln and "relaxed" not in ln and "cga_arrive" not in ln]
    assert not bad, f"{family}: cluster-scope arrive without relaxed=True: {bad}"


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_no_internal_ptxas_knobs(family):
    """The ptxas scheduling-pin pragma, its ptxas option and the pre-upstream DSL's scheduling-fence helper are
    debugging tools only: never in a production kernel (a scheduling pin is not a memory fence; an SMEM-to-async
    boundary takes ``fence_proxy``).  The three tokens are assembled from fragments so this source spells none of them."""
    src = _kernel_source(family)
    knobs = ["".join(p) for p in (("uu", "mn"), ("Fence", "Code"), ("cf", "ence"))]
    hits = sorted({m.group(0) for m in re.finditer("|".join(knobs), src)})
    assert not hits, f"{family}: an internal ptxas scheduling knob is in the kernel: {hits}"


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_kernel_is_named_by_geometry(family):
    """No pre-port project / model / DSL names in code or comments: the kernel is ``d256``, its provenance is the plan's."""
    src = _kernel_source(family)
    # The pre-port project / model / DSL names, assembled from fragments so this source never spells them either.
    words = ["".join(p) for p in (("vibe", "tile"), ("qw", "en"), ("dk", "g"), ("tile_", "ct", "m"), ("ct", "m"))]
    hits = sorted({m.group(0) for m in re.finditer(r"(?i)\b(" + "|".join(words) + r")\b", src)})
    assert not hits, f"{family}: pre-port names in the kernel source: {hits}"


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_idesc_k_dim_comes_from_the_config(family):
    """fp8 on Rubin is the K=64 path with idesc ``k_dim=1`` (Blackwell is the OPPOSITE); f16 is ``TILE_K_HW=16`` with the
    default ``k_dim=0`` (plan Q6, mma-tma-matrix s1-2).  A wrong pair silently scrambles accumulator ROWS -- no crash,
    passes a 1-tile shape -- so the value lives ONCE in config_sm107 (``IDESC_K_DIM``) and the body reads it; a
    ``k_dim=<literal>`` at a call site bypasses the validator."""
    code = _code_only(_kernel_source(family))
    spellings = sorted(set(re.findall(r"\bk_dim\s*=\s*([^,)\s]+)", code)))
    assert spellings, f"{family}: no idesc build site passes k_dim= (the config's IDESC_K_DIM never reaches the descriptors)"
    assert spellings == ["CFG.IDESC_K_DIM"], f"{family}: k_dim= must be spelled k_dim=CFG.IDESC_K_DIM at every build site; found {spellings}"


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_q_loop_bounds_come_from_the_tile_dsl_primitive(family):
    """The kv block's q-tile range (which q tiles attend a kv block under causal / SWA / bottom-right) is
    ``tile_dsl.mask.compute_q_loop_bounds`` (plan s4.14), not a per-kernel copy -- a private bound is how a mask arm
    silently attends one tile too many or too few.  A local ``_q_loop_bounds`` WRAPPER that delegates to it and adds the
    forced N >= 1 fully-masked-tile clamp at the call site is the intended shape.  (The per-cell P mask itself may be
    local: tile_dsl has no TRANSPOSED row = kv / col = q bit-word arm yet -- plan s13 PR-4.)"""
    code = _code_only(_kernel_source(family))
    assert re.search(
        r"\bcompute_q_loop_bounds\(", code
    ), f"{family}: the kernel never CALLS tile_dsl.mask.compute_q_loop_bounds (a private bound is a mask-arm hazard)"


# =========================================================================== SASS pins: trace-compile both bodies for sm_107a on ANY box
# What no numerics test can see: whether the 224 / 56 (f16) or 232 / 40 (fp8) register split reached the binary (ptxas C7508 drops every
# setmaxregister when it cannot determine the entry count -- the 8-warp d512 launches lose it; these 12-warp bodies must
# keep it), stack spills in the tight 40-register roles, a GPU-scope drain before a cluster arrive, and the TMEM-load /
# publish-arrive order (an arrive scheduled BETWEEN two LDTMs of one accumulator slot lets the parked MMA overwrite the
# slot under the pending second read: exactly the second half of the columns wrong, 2nd+ work items only).  One
# sm_107a trace-compile per row (~20-60 s; ``CUTE_DSL_ARCH`` needs no device).  SKIPS when the DSL predates sm_107a or no
# nvdisasm decodes the cubin; a compile failure -- or a compile() signature this probe cannot satisfy -- is a FAIL.
_SASS_PROBE = textwrap.dedent(r"""
    import glob, inspect, os, re, subprocess, sys
    dump, family, mask, cands = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "cubin"            # keep the cubin, disassemble it ourselves
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"          # unconditional: an inherited value would pin the wrong target's SASS
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"  # a compiled-plan cache HIT skips ptxas and dumps no cubin
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP32
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams
    FILES = {"f16": "sm107/bprop_d256_f16.py", "fp8": "sm107/bprop_d256_fp8.py", "mxfp8": "sm107/bprop_d256_mxfp8.py"}
    # *_f32dv: the fp8 body at dtype_o = FP32 -- the GQA fold's unrounded dV partial (the 128 KiB staging, two 32-col stores per chunk).
    MASKS = {"dense": {}, "causal": dict(window_right=0), "causal_swa": dict(window_right=0, window_left=640),
             "thd": dict(thd_varlen=True), "thd_causal": dict(thd_varlen=True, window_right=0),
             "dense_f32dv": dict(dtype_o=DTYPE_FP32), "causal_f32dv": dict(window_right=0, dtype_o=DTYPE_FP32),
             "thd_causal_f32dv": dict(thd_varlen=True, window_right=0, dtype_o=DTYPE_FP32)}
    params = TemplateParams(dtype_qkv=DTYPE_BF16 if family == "f16" else DTYPE_E4M3, **MASKS[mask])
    mod = load_template(_sm100_kernel_path(FILES[family]), params, tag="sdpa_bwd_sm107_main_" + family)
    # The shape arguments of compile(), by KEYWORD against its own signature (every spelling the plan and the forward
    # kernels use is offered: B=1, H=8, S=1024, d=256 -- 4 kv blocks x 8 q tiles); a REQUIRED parameter none of them
    # covers is a FAIL that names it, so the probe is extended deliberately rather than compiling the wrong thing.
    CAND = dict(b=1, batch=1, n_batch=1, qh=8, h_q=8, hq=8, heads_q=8, n_heads=8, qh_chunk=8, kh=8, h_kv=8, hkv=8, heads_kv=8,
                sq=1024, s_q=1024, seq_q=1024, s_q_pad=1024, skv=1024, s_kv=1024, seq_kv=1024, s_kv_pad=1024,
                d_qk=256, d_v=256, d=256, has_sink=False, has_amax=True, emit_amax=True)
    sig = inspect.signature(mod.compile).parameters
    kw = {k: v for k, v in CAND.items() if k in sig}
    missing = [n for n, p in sig.items() if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n not in kw]
    if missing:
        print("FAIL compile() has required parameters this probe does not know:", missing, "-- extend CAND in the test"); sys.exit(4)
    mod.compile(**kw)
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    if not cubins:
        print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    def cnt(*subs):
        return sum(1 for ln in sass if all(sb in ln for sb in subs))
    # The main kernel's section only: a THD build's cubin also carries the one-shot setup kernel (descriptor patching, whose
    # GENERIC->TENSORMAP release fence lowers to MEMBAR.ALL.GPU + CGAERRBAR), which the per-tile drain pin must not read.
    main_end = next((i for i, ln in enumerate(sass) if ".text." in ln and "setup_kernel" in ln), len(sass))
    main_sass = sass[:main_end]
    def cnt_main(*subs):
        return sum(1 for ln in main_sass if all(sb in ln for sb in subs))
    print("SASS USETMAXREG", cnt_main("USETMAXREG"))
    print("SASS STL", cnt_main("STL"))
    print("SASS LDL", cnt_main("LDL"))
    print("SASS MEMBAR_GPU", cnt_main("MEMBAR.ALL.GPU"))
    print("SASS CGAERRBAR", cnt_main("CGAERRBAR"))
    print("SASS SETUP_KERNEL_LINES", len(sass) - main_end)
    print("SASS SYNCS_PHASECHK", sum(1 for ln in sass if "SYNCS.PHASECHK" in ln and "USYNCS.PHASECHK" not in ln))
    print("SASS USYNCS_PHASECHK", cnt("USYNCS.PHASECHK"))
    print("SASS NANOSLEEP", cnt("NANOSLEEP"))
    print("SASS LINES", len(sass))
    print("SASS R2P", cnt(" R2P"))
    # TMEM-load / publish-arrive order (frost-kernels.md s3): in every innermost loop body that carries an exp2 burst (the
    # softmax recompute), the LDTMs of ONE accumulator slot -- same base register, same 128-column window -- must not
    # have an ARRIVE scheduled between their first and their last.
    INS = re.compile(r"^\s+/\*([0-9a-f]+)\*/\s+(?:(@!?U?P[0-9T]+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;")
    LABEL = re.compile(r"^(\.L_x_\d+):")
    ins, labels, pending = [], {}, []
    for ln in sass:
        m = LABEL.match(ln)
        if m:
            pending.append(m.group(1)); continue
        m = INS.match(ln)
        if not m:
            continue
        for lab in pending:
            labels[lab] = len(ins)
        pending = []
        ins.append((m.group(3), m.group(4)))
    bodies = []
    for i, (op, args) in enumerate(ins):
        if not op.startswith("BRA"):
            continue
        m = re.search(r"(\.L_x_\d+)", args)
        if m and m.group(1) in labels and labels[m.group(1)] <= i:
            t = labels[m.group(1)]
            body = ins[t : i + 1]
            if len(body) >= 40 and sum(1 for o, _ in body if o.startswith("MUFU.EX2")) >= 16:
                bodies.append((t, i))
    inner = [(t, e) for t, e in bodies if not any((t2, e2) != (t, e) and t <= e2 <= e for t2, e2 in bodies)]
    n_groups = n_bad = n_ldtm = 0
    for t, e in inner:
        slots = {}
        arrives = []
        for j in range(t, e + 1):
            op, args = ins[j]
            if op.startswith("LDTM"):
                n_ldtm += 1
                m = re.search(r"tmem\[(U?R\d+)(?:\+(0x[0-9a-fA-F]+))?\]", args)
                if m:
                    slots.setdefault((m.group(1), int(m.group(2) or "0", 16) // 0x80), []).append(j)
            elif "ARRIVE" in op:
                arrives.append(j)
        for key, idx in slots.items():
            if len(idx) < 2:
                continue
            n_groups += 1
            lo, hi = min(idx), max(idx)
            between = [a for a in arrives if lo < a < hi]
            if between:
                n_bad += 1
                print("LDTM_ORDER_VIOLATION body", hex(t), "slot", key, "ldtm", idx, "arrive", between)
    print("LDTM_ORDER_BODIES", len(inner))
    print("LDTM_ORDER_LDTMS", n_ldtm)
    print("LDTM_ORDER_GROUPS", n_groups)
    print("LDTM_ORDER_VIOLATIONS", n_bad)
    """)

_SASS_PIN_ROWS = [
    pytest.param("f16", "dense", id="f16-dense"),
    pytest.param("f16", "causal", id="f16-causal"),
    # The f16 body's THD arms (packed-total-clamped runtime descriptors, the device claim counter, the q band, the per-sequence
    # dV descriptors; MASK_PADDED set, so the bit-word mask arm is live on the "dense" THD row too).  The probe's cubin also holds
    # the one-shot setup kernel, whose GENERIC->TENSORMAP release fence is one MEMBAR.ALL.GPU + CGAERRBAR -- the drain pin below
    # counts the MAIN kernel only (`test_sm107_register_split_spills_and_drains_sass_pins`).
    pytest.param("f16", "thd", id="f16-thd"),
    pytest.param("f16", "thd_causal", id="f16-thd-causal"),
    pytest.param("fp8", "dense", id="fp8-dense"),
    pytest.param("fp8", "causal", id="fp8-causal"),
    # The sliding-window specialization the fp8 SWA perf cell runs: `compute_q_loop_bounds` trims each kv block's q loop
    # from ABOVE by the window (the same call the f16 body makes), the SWA term joins the causal one in the bit-word mask
    # arm.  Its SASS is the causal row's plus the trim arithmetic (+32 lines, every other pinned count identical, 2026-09-28).
    pytest.param("fp8", "causal_swa", id="fp8-causal-swa"),
    # The fp8 body's THD arms (the f16 mechanism in e4m3: packed-total-clamped runtime descriptors, the device claim counter, the
    # THD q band, the per-sequence clipped dV descriptors, the amax row gate over the live region); the cubin carries the one-shot
    # setup kernel too, counted out of the drain pin as on the f16 rows.
    pytest.param("fp8", "thd", id="fp8-thd"),
    pytest.param("fp8", "thd_causal", id="fp8-thd-causal"),
    # The fp8 body the fp8 adapter loads under GQA since the fold rounds once: dtype_o = FP32, the per-Q-head dV_true stored
    # UNROUNDED through a 128 KiB staging (322 KiB of slabs, 1 KiB under the usable cap) as two 32-column halves per epilogue chunk --
    # the same pins as the bf16-out rows (no spill, no drain, the bit-word mask arm, the LDTM order), dense / causal / THD causal.
    pytest.param("fp8", "dense_f32dv", id="fp8-dense-f32dv"),
    pytest.param("fp8", "causal_f32dv", id="fp8-causal-f32dv"),
    pytest.param("fp8", "thd_causal_f32dv", id="fp8-thd-causal-f32dv"),
    # The MXFP8 body at the bare record (the FMUL arm of the P quantizer, the block-scaled dS default P-b): the fp8 rows' pins
    # hold for it too; its own arms (fused scaled cvt, MASK_Q_PAD, the bf16-dS twin) are pinned in test_sdpa_bwd_mxfp8_sm107.py.
    pytest.param("mxfp8", "dense", id="mxfp8-dense"),
    pytest.param("mxfp8", "causal", id="mxfp8-causal"),
]
# The masked rows of the above: the mask form pin (rules/frost-tile-dsl.md s10d) applies to them only.
_MASKED_SASS_PIN_ROWS = [r for r in _SASS_PIN_ROWS if not r.values[1].startswith("dense")]
# Spill bounds: the counts MEASURED on the branch's own toolchain (cutlass-dsl 4.8.0 + the internal CUDA toolkit's ptxas,
# 2026-09-23, B=1 H=8 S=1024; the fp8 SWA row 2026-09-28): 0 / 0 STL / LDL on every row -- the plan's target (s10.9) --
# plus the DSL / ptxas jitter frost_test_utils.SPILL_TOLERANCE allows.  Never loosen a row to turn it green; a real spill
# adds tens.
_SPILL_PINS = {
    ("f16", "dense"): {"STL": 0, "LDL": 0},
    ("f16", "causal"): {"STL": 0, "LDL": 0},
    # THD rows, MEASURED at the port (2026-10-01, B=1 H=8 S=1024 packed): the dense THD arm 0 / 0; the causal THD arm spills
    # 3 STL / 10 LDL in the 56-register service warps (the MMA warp's per-sequence bounds, one value re-read in the scheduler
    # warp's stats loop) -- a bring-up finding to fix by hoisting, recorded here as the measured bound, never to be loosened.
    ("f16", "thd"): {"STL": 0, "LDL": 0},
    ("f16", "thd_causal"): {"STL": 3, "LDL": 10},
    ("fp8", "dense"): {"STL": 0, "LDL": 0},
    ("fp8", "causal"): {"STL": 0, "LDL": 0},
    ("fp8", "causal_swa"): {"STL": 0, "LDL": 0},
    # The fp8 THD arms, MEASURED at the port (2026-10-02, B=1 H=8 S=1024 packed): an 8-byte frame -- one STL at kernel entry (a
    # kernel-invariant fp32 scale product) and LDLs in the 232-register softmax warps (one per q iteration + two once per role);
    # every dense / causal / SWA / bottom-right / padded arm stays 0 / 0 and instruction-identical.  A bring-up bound to fix by
    # hoisting (the 224 / 56 split would take registers FROM the spilling warps), recorded here, never to be loosened.
    ("fp8", "thd"): {"STL": 1, "LDL": 3},
    ("fp8", "thd_causal"): {"STL": 1, "LDL": 3},
    # The fp32-dV rows (2026-10-06, B=1 H=8 S=1024, the same toolchain): the dense / causal builds 0 / 0 like their bf16-out twins
    # (USETMAXREG 5, 9 UTMASTG: eight 32-col dV subtiles + the dS slot); the THD causal build the fp8 THD arm's 1 / 3 frame.
    ("fp8", "dense_f32dv"): {"STL": 0, "LDL": 0},
    ("fp8", "causal_f32dv"): {"STL": 0, "LDL": 0},
    ("fp8", "thd_causal_f32dv"): {"STL": 1, "LDL": 3},
    ("mxfp8", "dense"): {"STL": 0, "LDL": 0},  # 2026-09-30: REG 168, 0 / 0 on every arm
    ("mxfp8", "causal"): {"STL": 0, "LDL": 0},
}
# One trace-compile per (family, mask) per session: every pin below reads the same SASS, so the probe runs once and
# the tests share its counts (a compile is 20-60 s; the dump dir of the FIRST caller holds the cubin).
_SASS_CACHE = {}


def _sass_probe(tmp_path, family, mask):
    if (family, mask) in _SASS_CACHE:
        return _SASS_CACHE[(family, mask)]
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    _kernel_source(family)
    dump = tmp_path / f"sm107a_bwd_{family}_{mask}"
    dump.mkdir()
    argv = [sys.executable, "-c", _SASS_PROBE, str(dump), family, mask, *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the {family} {mask} backward failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}
    order = {ln.split()[0]: int(ln.split()[1]) for ln in out if ln.startswith("LDTM_ORDER_") and len(ln.split()) == 2}
    print(f"\nsm107 bwd {family} {mask} sm_107a SASS: {stats}; {order}; {[ln for ln in out if ln.startswith('LDTM_ORDER_VIOLATION ')]}")
    _SASS_CACHE[(family, mask)] = (stats, order)
    return stats, order


@pytest.mark.parametrize("family, mask", _SASS_PIN_ROWS)
def test_sm107_register_split_spills_and_drains_sass_pins(tmp_path, family, mask):
    """USETMAXREG > 0 (the register split is real in the binary, not dropped by C7508), no new stack spills, and no
    GPU-scope drain (MEMBAR.ALL.GPU == CGAERRBAR == 0) on a per-tile path."""
    stats, _order = _sass_probe(tmp_path, family, mask)
    assert stats["USETMAXREG"] > 0, "no USETMAXREG: ptxas dropped the register split (C7508 -- the entry register count is undetermined)"
    assert_no_new_spills(stats, _SPILL_PINS[(family, mask)], tag=f"{family} {mask}: ")
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is on a per-tile path (GPU-scope drain)"


@pytest.mark.parametrize("family, mask", _SASS_PIN_ROWS)
def test_sm107_every_tmem_load_precedes_the_arrive_that_frees_its_slot(tmp_path, family, mask):
    """No arrive is scheduled BETWEEN two LDTMs of one accumulator slot (frost-kernels.md s3; the sm107 d512 forward's
    masked arms lacked the ``tcgen05_wait(LOAD)`` that orders it until 6b629d1d).  On these bodies each softmax warp
    reads a slot with ONE ``LDTM.x64`` (the two warpgroups split the 128-column q tile) followed by ``NOP`` and the
    ``USYNCS.ARRIVE`` that frees it, so the two-chunk hazard shape has nothing to bite today -- the pin guards the shape
    the moment a body reads a slot in two chunks (a 128-column per-warp read, the dV epilogue moving into the loop).
    The detector must have found the softmax body AND its accumulator loads, or it pins nothing."""
    _stats, order = _sass_probe(tmp_path, family, mask)
    assert (
        order.get("LDTM_ORDER_BODIES", 0) > 0
    ), "the detector found no per-iteration body with an exp2 burst -- it is not pinning anything; adjust its body filter"
    assert order.get("LDTM_ORDER_LDTMS", 0) > 0, "the detector saw no LDTM inside the softmax body -- its operand parse or body filter needs a look"
    assert (
        order["LDTM_ORDER_VIOLATIONS"] == 0
    ), "an ARRIVE is scheduled between two LDTMs of one accumulator slot (missing tcgen05_wait(LOAD) before the publish)"


@pytest.mark.parametrize("family, mask", _MASKED_SASS_PIN_ROWS)
def test_sm107_masked_arm_lowers_to_the_bit_word_form(tmp_path, family, mask):
    """A masked softmax arm must be the bit-word form -- one keep-word per 32 columns, ``R2P`` + one ``FSEL`` per cell --
    never the per-cell compare + select (rules/frost-tile-dsl.md s10d: 2-3x the dense tile's instruction count; the f16
    body's causal build read ISETP 94 / FSEL 64 / R2P 0 before it took the fp8 body's ``band_mask_words`` arm, 54 / 64 / 8
    after).  ``R2P == 0`` on a masked build is the detector the rule names."""
    stats, _order = _sass_probe(tmp_path, family, mask)
    assert stats["R2P"] > 0, f"{family} {mask}: no R2P in the masked build -- the mask arm is the per-cell compare + select form"


@pytest.mark.parametrize("family", _FAMILIES)
def test_sm107_every_mask_site_is_the_bit_word_arm(family):
    """The mask op has ONE form (PR #1209 retired ``MASK_FORM`` / ``apply_mask_chunk_form`` / ``apply_mask_chunk_bits``):
    the kv-major bodies spell it as ``band_mask_words`` + ``apply_mask_words`` (the transpose of ``apply_mask_chunk``),
    and carry none of the retired names -- a per-cell compare + select arm reintroduced under a local constant would
    pass the SASS pin on the family it was not enabled for."""
    code = _code_only(_kernel_source(family))
    for gone in ("MASK_FORM", "apply_mask_chunk_form", "apply_mask_chunk_bits", "form="):
        assert gone not in code, f"{family}: {gone!r} is a retired mask-form spelling"
    assert "band_mask_words(" in code and "apply_mask_words(" in code, f"{family}: the masked arm must be the library's bit-word primitives"
    assert "arith.select(" in code, f"{family}: the padded arm's band bound is the only select left; a per-cell select loop is the old form"


# =========================================================================== SASS pins: the d256 stage-3 GEMM rendering for sm_107a on ANY box
# The rendering is a fork of the generic GEMM template at constants no other engine renders (cluster 2x1, 128 x 128 per
# CTA, 6 stages, 2 accumulator stages), so the toolchain facts the stage-2 pins guard are re-checked on it: no stack
# spill in the 24-register producer warps, no GPU-scope drain before a cluster arrive (every cluster-scope arrive here is
# `relaxed`), and the version-0 tcgen05 SMEM descriptors reaching every operand ring root.  `mod._host` is wrapped in the
# pointer-argument @cute.jit the prepared hosts use (`compile_host_f16` builds the same `make_ptr` operands), so the
# device cubin is the artifact's.  One sm_107a trace-compile per row (~5-10 s).
_GEMM_SASS_PROBE = textwrap.dedent(r"""
    import glob, os, subprocess, sys
    dump, major, mask, arm, cands = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass
    import cutlass.cute as cute
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.api_dsl_sm107 import _stage3_cgrp_tile_mn, _stage3_params
    from cudnn.sdpa.bwd.config_sm100 import EPI_DESCALE, EPI_NONE, EPI_QUANT
    # The fp8 arm's records exactly as the fp8 adapter builds them (GQA: dK DESCALE, dQ QUANT; MHA: both QUANT).
    ARMS = {"bf16": (DTYPE_BF16, (EPI_NONE, EPI_NONE), -1), "fp8-descale": (DTYPE_E4M3, (EPI_DESCALE, EPI_QUANT), DTYPE_E4M3),
            "fp8-quant-e4m3": (DTYPE_E4M3, (EPI_QUANT, EPI_QUANT), DTYPE_E4M3), "fp8-quant-bf16": (DTYPE_E4M3, (EPI_QUANT, EPI_QUANT), DTYPE_BF16)}
    ds_code, epi_modes, dtype_out = ARMS[arm]
    p_dk, p_dq = _stage3_params(ds_code, causal=(mask == "causal"), shift=0, gran=256, trim=True, cgrp_tile_mn=_stage3_cgrp_tile_mn(107, 256), epi_modes=epi_modes, dtype_out=dtype_out)
    params = p_dq if major == "dq" else p_dk
    mod = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), params, tag="sass_probe_stage3_" + major + "_" + arm)
    print("PARAMS", repr(params))
    for name in ("cgrp_tile_mnk", "cta_tile_mnk", "cluster_shape_mnk", "ab_stages", "acc_stages", "mma_size_m", "multicast_a", "a_mcast_slices", "ab_empty_full_mask",
                 "mma_inst_shape_mnk", "mma_k_dim", "mma_size_k", "epi_mode", "_EPI_ROW_BYTES"):
        print("CONST", name, repr(getattr(mod, name)))
    print("CONST mma_kind", str(mod.mma_kind))
    layout = mod._smem_layout_bytes()
    for name in ("smem_a_0", "smem_b_0", "smem_d", "total"):
        print("SMEM", name, layout[name])
    S_Q, S_KV, H, B, D = 1024, 1024, 8, 1, 256
    a_m_major = bool(params.a_is_m_major)
    epi = int(params.epi_mode)

    @cute.jit
    def probe(entry: cutlass.Constexpr, a_ptr: cute.Pointer, b_ptr: cute.Pointer, c_ptr: cute.Pointer, meta_ptr: cute.Pointer, desc_ptr: cute.Pointer,
              d0_ptr: cute.Pointer, d1_ptr: cute.Pointer, s_ptr: cute.Pointer, amax_ptr: cute.Pointer, stream):
        if cutlass.const_expr(a_m_major):
            a = cute.make_tensor(a_ptr, cute.make_layout((S_Q, S_KV, H, B), stride=(1, S_Q, S_KV * S_Q, H * S_KV * S_Q)))
            b = cute.make_tensor(b_ptr, cute.make_layout((D, S_KV, H, B), stride=(1, H * D, D, S_KV * H * D)))
            c = cute.make_tensor(c_ptr, cute.make_layout((S_Q, D, H, B), stride=(H * D, 1, D, S_Q * H * D)))
        else:
            a = cute.make_tensor(a_ptr, cute.make_layout((S_KV, S_Q, H, B), stride=(S_Q, 1, S_KV * S_Q, H * S_KV * S_Q)))
            b = cute.make_tensor(b_ptr, cute.make_layout((D, S_Q, H, B), stride=(1, H * D, D, S_Q * H * D)))
            c = cute.make_tensor(c_ptr, cute.make_layout((S_KV, D, H, B), stride=(H * D, 1, D, S_KV * H * D)))
        meta = cute.make_tensor(meta_ptr, cute.make_layout((1,), stride=(1,)))
        desc = cute.make_tensor(desc_ptr, cute.make_layout((1,), stride=(1,)))
        problem = tuple(cutlass.Int64(x) for x in (a.shape[0], b.shape[0], a.shape[1], H, B, *a.stride, *b.stride, *c.stride, b.shape[1], a.shape[0], c.shape[0]))
        if cutlass.const_expr(epi == EPI_NONE):
            entry(problem, a, b, c, meta, desc, stream)  # the SM100 chain's positional seven arguments
        else:
            sc = cute.make_layout((1,), stride=(1,))
            s_out = cute.make_tensor(s_ptr, sc) if cutlass.const_expr(epi == EPI_QUANT) else None
            entry(problem, a, b, c, meta, desc, stream, cute.make_tensor(d0_ptr, sc), cute.make_tensor(d1_ptr, sc), s_out, cute.make_tensor(amax_ptr, sc) if cutlass.const_expr(epi == EPI_QUANT) else None)

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    ab_dt, cd_dt = mod.ab_dtype, mod.cd_dtype
    cute.compile(probe, mod._host, ptr(ab_dt), ptr(ab_dt), ptr(cd_dt), ptr(cutlass.Int32, 4), ptr(cutlass.Int64, 8),
                 ptr(cutlass.Float32, 4), ptr(cutlass.Float32, 4), ptr(cutlass.Float32, 4), ptr(cutlass.Float32, 4),
                 cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi --gpu-arch sm_107a")
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    if not cubins:
        print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    def cnt(*subs):
        return sum(1 for ln in sass if all(sb in ln for sb in subs))
    print("SASS STL", cnt("STL"))
    print("SASS LDL", cnt("LDL"))
    print("SASS MEMBAR_GPU", cnt("MEMBAR.ALL.GPU"))
    print("SASS CGAERRBAR", cnt("CGAERRBAR"))
    print("SASS UTCMMA", cnt("UTCMMA") + cnt("UTCHMMA"))
    print("SASS UTCQMMA", cnt("UTCQMMA"))
    print("SASS FMNMX3", cnt("FMNMX3"))
    print("SASS FSEL", cnt(" FSEL"))
    print("SASS REDG_MAX", cnt("REDG.E.MAX"))
    print("SASS F2FP", cnt("F2FP"))
    print("SASS LINES", len(sass))
    """)

_GEMM_SASS_ROWS = [
    pytest.param("dk", "dense", "bf16", id="dk-dense"),
    pytest.param("dq", "dense", "bf16", id="dq-dense"),
    pytest.param("dk", "causal", "bf16", id="dk-causal"),
    pytest.param("dq", "causal", "bf16", id="dq-causal"),
    # The fp8 arm as the fp8 adapter renders it: the GQA records (dK DESCALE -> bf16 partials, dQ QUANT -> e4m3 dQ), the MHA
    # records (both QUANT -> e4m3) and the half-gradient variant (QUANT -> bf16 with scale 1.0).
    pytest.param("dk", "causal", "fp8-descale", id="dk-causal-fp8-descale"),
    pytest.param("dq", "causal", "fp8-descale", id="dq-causal-fp8-quant-e4m3"),
    pytest.param("dk", "dense", "fp8-quant-e4m3", id="dk-dense-fp8-quant-e4m3"),
    pytest.param("dq", "dense", "fp8-quant-bf16", id="dq-dense-fp8-quant-bf16"),
]
# MEASURED on the branch's toolchain (cutlass-dsl 4.8.0 + the internal CUDA 13.5 ptxas, 2026-09-28): 0 / 0 on every row
# (the fp8 arm's epilogue warps read REG 90-116 with the amax fold in registers).
_GEMM_SPILL_PINS = {"STL": 0, "LDL": 0}
_GEMM_SASS_CACHE = {}


def _gemm_sass_probe(tmp_path, major, mask, arm="bf16"):
    if (major, mask, arm) in _GEMM_SASS_CACHE:
        return _GEMM_SASS_CACHE[(major, mask, arm)]
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"sm107a_bwd_stage3_{major}_{mask}_{arm}"
    dump.mkdir()
    # From a FILE, not `-c`: the probe defines a `@cute.jit` wrapper and the DSL needs its source (`UNSUP_NO_SOURCE` otherwise).
    script = dump / "gemm_sass_probe.py"
    script.write_text(_GEMM_SASS_PROBE)
    proc = subprocess.run([sys.executable, str(script), str(dump), major, mask, arm, *cands], capture_output=True, text=True, timeout=900)
    assert (
        proc.returncode == 0
    ), f"sm_107a trace-compile of the d256 stage-3 {major} {mask} {arm} rendering failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}
    consts = {ln.split()[1]: ln.split(maxsplit=2)[2] for ln in out if ln.startswith("CONST ")}
    smem = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SMEM ")}
    print(f"\nsm107 bwd stage-3 {major} {mask} {arm} sm_107a SASS: {stats}; {consts}; SMEM {smem}")
    _GEMM_SASS_CACHE[(major, mask, arm)] = (stats, consts, smem)
    return stats, consts, smem


@pytest.mark.parametrize("major, mask, arm", _GEMM_SASS_ROWS)
def test_stage3_d256_rendering_sass_pins(tmp_path, major, mask, arm):
    """The shipped d256 renderings (what `_stage3_cgrp_tile_mn(107, 256)` selects, bf16 and the fp8 arm), trace-compiled
    for sm_107a: the (256, 256) row's constants reached the module, no stack spills, no GPU-scope drain on a per-tile path,
    and both MMA-operand ring roots sit below 256 KiB -- so the version-0 tcgen05 SMEM descriptor is the right one at this
    ring depth (a 9-stage Rubin row would put the B ring's UPPER stages past 256 KiB while its root stays below: the
    per-stage advance is a bare add on the root, which is exactly why the ROOT is what is pinned).  The fp8 arm additionally:
    the K64 form (256x256x64, k_dim 1, two `UTCQMMA` k-blocks per stage -- the dense-FP8 MMA mnemonic), its QUANT epilogue's
    amax fold on `FMNMX3` (never compare + select: no FSEL in any epilogue) with ONE per-warp `REDG.E.MAX` atomic site, and
    the fp8 / bf16 output converts (`F2FP`) -- while the DESCALE epilogue (the GQA dK partial) stores the fp32 true-unit value
    UNCONVERTED (no `F2FP`: the fold rounds the group's sum once) through the 32-element 128-B staging row."""
    stats, consts, smem = _gemm_sass_probe(tmp_path, major, mask, arm)
    fp8 = arm != "bf16"
    quant = arm.startswith("fp8-quant") or (arm == "fp8-descale" and major == "dq")
    assert consts["cgrp_tile_mnk"] == ("(256, 256, 128)" if fp8 else "(256, 256, 64)") and consts["cluster_shape_mnk"] == "(2, 1, 1)", consts
    assert consts["ab_stages"] == "6" and consts["acc_stages"] == "2" and consts["mma_size_m"] == "1", consts
    assert consts["multicast_a"] == "False" and consts["a_mcast_slices"] == "1" and consts["ab_empty_full_mask"] == "False", consts
    if fp8:
        assert consts["mma_inst_shape_mnk"] == "(256, 256, 64)" and consts["mma_k_dim"] == "1" and consts["mma_size_k"] == "2", consts
        assert "f8f6f4" in consts["mma_kind"].lower(), consts
        assert stats["UTCQMMA"] == 2 and stats["UTCMMA"] == 0, "the fp8 arm issues the dense-FP8 MMA (two K64 blocks per stage)"
        if quant:
            assert stats["F2FP"] > 0, "the QUANT epilogue converts the descaled fp32 accumulator to the gradient dtype"
            assert stats["FMNMX3"] > 0 and stats["REDG_MAX"] == 1, "the QUANT epilogue's amax fold (max.f32 -> FMNMX3) and its ONE per-warp atomicMax site"
        else:
            assert stats["F2FP"] == 0, "the DESCALE epilogue stores the fp32 per-Q-head partial unconverted (the GQA fold rounds the group's sum once)"
            assert stats["FMNMX3"] == 0 and stats["REDG_MAX"] == 0, "DESCALE has no amax fold and no atomic"
        assert consts["_EPI_ROW_BYTES"] == ("64" if arm == "fp8-quant-e4m3" or (arm == "fp8-descale" and major == "dq") else "128"), consts
    else:
        assert consts["mma_inst_shape_mnk"] == "(256, 256, 16)" and consts["mma_k_dim"] == "0" and consts["mma_size_k"] == "4", consts
        assert stats["UTCQMMA"] == 0 and stats["FMNMX3"] == 0 and stats["REDG_MAX"] == 0, "the bf16 rows carry no fp8 MMA, no amax fold, no atomicMax"
    assert stats["FSEL"] == 0, "no compare + select in an epilogue (a cute.math.max would show as FSEL)"
    assert_no_new_spills(stats, _GEMM_SPILL_PINS, tag=f"stage-3 {major} {mask} {arm}: ")
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is on a per-tile path (GPU-scope drain)"
    assert smem["smem_a_0"] < _SMEM_DESC_V0_LIMIT and smem["smem_b_0"] < _SMEM_DESC_V0_LIMIT, smem
    assert smem["total"] <= _SM100_OPTIN_SMEM, smem


# =========================================================================== the stage-2 dS workspace budget (host)


def test_every_sm107_row_chunks_against_the_one_8_gib_budget():
    """ONE stage-2 dS workspace budget for every sm107 row and dS policy -- ``_SM107_WS_BUDGET_BYTES``, 8 GiB, no per-row or
    per-policy constant (the MXFP8 row used to carry its own) and no adapter override of ``_ws_budget_bytes``.  A chunking
    constant only: the plan's workspace is the chunk's whole carve.  The arithmetic its comment states, pinned at B=1 H=128:
    8K bf16 / fp16 dS (2 B per element) 64 heads = 2 launches, e4m3 dS (1 B) 128 heads = 1 launch; what 4 GiB gave next to it
    (32 / 64 heads); at 16K every row still chunks (16 / 32 heads).  Rubin (cc 10.7, 212 SMs, SM clock 2376 MHz) measured the
    head-chunk cost at that shape: the bf16-dS chain forced from 32-head to 16-head chunks lost 0.9 % dense / 6.6 % causal."""
    import inspect

    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107
    from cudnn.sdpa.bwd.api_dsl import _SM100_WS_BUDGET_BYTES
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107, SdpaBwdDslSm107Fp8, SdpaBwdDslSm107Mxfp8, _sm107_chunks

    assert sm107._SM107_WS_BUDGET_BYTES == 8 << 30 == 2 * _SM100_WS_BUDGET_BYTES, "8 GiB on the Rubin line; the SM100 chain keeps its 4 GiB"
    assert not hasattr(sm107, "_SM107_MXFP8_BLOCK_SCALED_WS_BUDGET_BYTES"), "one budget, one name"
    for cls in (SdpaBwdDslSm107Fp8, SdpaBwdDslSm107Mxfp8):
        assert "_ws_budget_bytes" not in vars(cls), f"{cls.__name__} overrides the budget: one source of truth"
    assert inspect.signature(_sm107_chunks).parameters["budget"].default == sm107._SM107_WS_BUDGET_BYTES
    e4m3 = torch.float8_e4m3fn
    half = _adapter(SdpaBwdDslSm107, b=1, hq=128, hkv=128, sq=8192, skv=8192)
    fp8 = _adapter(SdpaBwdDslSm107Fp8, b=1, hq=128, hkv=128, sq=8192, skv=8192, dt=e4m3, grad_dt=e4m3)
    assert half._ws_budget_bytes() == fp8._ws_budget_bytes() == 8 << 30
    assert (half._b_chunk, half._qh_chunk) == (1, 64), "bf16 dS at 8K H=128: 64-head chunks, 2 launches"
    assert (fp8._b_chunk, fp8._qh_chunk) == (1, 128), "e4m3 dS at 8K H=128: the whole head set, 1 launch"
    assert half.scratch_workspace_bytes() >= 64 * 8192 * 8192 * 2 and fp8.scratch_workspace_bytes() >= 128 * 8192 * 8192, "the carve holds the chunk"
    assert _sm107_chunks(1, 128, 1, 8192, 8192, 2, budget=4 << 30) == (1, 32) and _sm107_chunks(
        1, 128, 1, 8192, 8192, 1, budget=4 << 30, batch_chunking=False
    ) == (1, 64)
    assert _sm107_chunks(1, 128, 1, 16384, 16384, 2) == (1, 16) and _sm107_chunks(1, 128, 1, 16384, 16384, 1, batch_chunking=False) == (
        1,
        32,
    ), "16K still chunks"
    assert _sm107_chunks(2, 128, 1, 16384, 16384, 2) == (2, 8), "heads shrink first: the batch stays whole while any head chunk fits at the full batch"


def test_plan_declines_typed_when_the_workspace_exceeds_what_the_caller_can_hold(monkeypatch):
    """The budget is a CHUNKING constant: the plan reports the chunk's whole carve through ``get_workspace_size()`` and the caller
    allocates it -- there is no adapter-side device-memory refusal.  A caller that cannot hold it bounds the ranked list with
    ``deselect_workspace_greater_than(free)`` (the free-memory query is the caller's; mocked here), and the pinned row then declines
    TYPED at build time -- ``cudnnGraphNotSupportedError`` naming the bytes and the bound -- before any launch; it never runs under
    a smaller carve.  The same graph with the bound at the requirement builds, and reports exactly the adapter's carve.  Host-only:
    the analyzer's device is faked to cc 10.7 and the adapter's JIT ``compile`` is a no-op (the decision needs the SIZE, not a
    kernel), so no kernel is traced and nothing can launch."""
    from cudnn.sdpa import _plan as plan_mod
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)  # host-only: the descriptor helper asks for the device even on a CUDA-less runner
    monkeypatch.setattr(SdpaBwdDslSm107, "compile", lambda self: None)  # size only: _prepared stays None, the plan's bytes come from scratch_workspace_bytes()

    def _never(*_a, **_k):
        raise AssertionError("a declined plan must not launch")

    monkeypatch.setattr(plan_mod._FrostSdpaPlan, "execute", _never)
    monkeypatch.setattr(SdpaBwdDslSm107, "execute", _never)
    shape = dict(b=1, hq=2, hkv=2, sq=512, skv=512)
    need = _adapter(SdpaBwdDslSm107, **shape).scratch_workspace_bytes()
    assert need > 0
    # the caller's free-memory query says less than the carve -> typed decline, no launch
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (need - 1, 32 << 30))
    g, _t, _outs = _build_graph(dt=torch.bfloat16, scale="default", **shape)
    select_engine(g, _ENGINE)
    free, _total = torch.cuda.mem_get_info()
    g.deselect_workspace_greater_than(free)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=rf"needs {need} workspace bytes, over the {free} limit"):
        g.build_plans()
    # the bound at the requirement: the plan builds and reports the adapter's carve, byte for byte
    g2, _t2, _outs2 = _build_graph(dt=torch.bfloat16, scale="default", **shape)
    select_engine(g2, _ENGINE)
    g2.deselect_workspace_greater_than(need)
    g2.build_plans()
    assert g2.get_workspace_size() == need


# =========================================================================== the d512 2x2 stage-2 twin's Rubin arm (host-only)
# The SM100 d512 backward's 2x2 stage 2 (``kernels/sm100/bprop_d512_f16_2x2.py``) is also the first FROST d512 backward
# that FITS Rubin: the same kernel at ``stages_kv=8, cast_stages=2`` fills the 325 KiB SMEM with an 8-stage K / V chunk
# ring.  No engine row serves cc 10.7 yet (that is a Capabilities change -- a SUPPORT_MATRIX_TRACKER.md edit in its own
# PR, Rule S2); what lands here is the arm's resource pins and its sm_107a trace-compile (Rule S6: a kernel feature lands on
# every arch line's test file, and its other-arch lowerings are smoke-compiled from whatever GPU you have).


def test_stage2_2x2_rubin_arm_config_pins():
    from cudnn.sdpa.bwd.config_sm100 import (
        SM107_USABLE_DYN_SMEM_2X2,
        TemplateParams2x2,
        desc_roots_2x2,
        desc_version_2x2,
        make_cfg_d512_2x2,
        smem_bytes_2x2,
        smem_layout_2x2,
        tmem_cols_2x2,
    )
    from cudnn.sdpa.bwd.config_sm107 import SMEM_CAP_BYTES, SMEM_SCAFFOLD_BYTES, TCGEN05_V0_ADDR_LIMIT

    cfg = make_cfg_d512_2x2(TemplateParams2x2(stages_kv=8, cast_stages=2, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2))
    assert (cfg.STAGES_KV, cfg.CAST_STAGES, cfg.D_CHUNK) == (8, 2, 64)
    assert smem_bytes_2x2(cfg) == 320 * 1024 and cfg.SMEM_CAP_BYTES == 325 * 1024 == SMEM_CAP_BYTES - SMEM_SCAFFOLD_BYTES
    assert tmem_cols_2x2(cfg) == 256 <= 512  # no is_exclusive needed: the public-wheel fence stays out of play
    # The zero-margin rule: sRingV stage 7 is the last descriptor root at 253952, its last byte 262143 = the v0 window's
    # last byte, and only because the cast slabs (TMA-store sources, no tcgen05 descriptor) are declared after the rings.
    assert [s.name for s in smem_layout_2x2(cfg)] == ["sQ", "sdO", "sRingK", "sRingV", "sCastS", "sCastDS"]
    assert max(off for _, off in desc_roots_2x2(cfg)) == 253952 and 253952 + 8192 == TCGEN05_V0_ADDR_LIMIT
    assert desc_version_2x2(cfg) == 0
    assert desc_version_2x2(make_cfg_d512_2x2(TemplateParams2x2(stages_kv=9, cast_stages=1, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2))) == 1


@pytest.mark.parametrize("mask", ["dense", "causal"])
def test_stage2_2x2_rubin_arm_trace_compiles_for_sm_107a(tmp_path, mask):
    """Rule S6: the Rubin arm's lowering, trace-compiled for sm_107a on any box (the DSL needs no device; skips where the
    wheel predates sm_107a).  The compiled rendering reports DESC_VERSION 0 and the 256-row cluster span."""
    from test_sdpa_bwd_dsl_sm100 import _STAGE2_PTX_PROBE
    from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a")
    dump = tmp_path / f"sm107a_stage2_2x2_{mask}"
    dump.mkdir()
    script = dump / "ptx_probe.py"
    script.write_text(_STAGE2_PTX_PROBE)
    kw = dict(dtype_qkv=2, kernel_file="sm100/bprop_d512_f16_2x2.py", twin=True, stages_kv=8, cast_stages=2, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)
    if mask == "causal":
        kw["window_right"] = 0
    import json

    proc = subprocess.run([sys.executable, str(script), str(dump), "sm_107a", json.dumps(kw)], capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the 2x2 stage-2 Rubin arm ({mask}) failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = dict(ln.split(maxsplit=1) for ln in proc.stdout.splitlines() if ln.startswith(("PTX_MD5", "CLUSTER_Q_ROWS", "DESC_VERSION", "N_CHUNKS")))
    assert out["DESC_VERSION"] == "0" and out["CLUSTER_Q_ROWS"] == "256" and out["N_CHUNKS"] == "8", out
    print(f"\nRubin 2x2 stage-2 {mask}: PTX md5 {out['PTX_MD5']}")
