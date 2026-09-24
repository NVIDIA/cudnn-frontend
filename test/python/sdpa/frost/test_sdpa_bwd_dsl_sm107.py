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
from pathlib import Path

import pytest
import torch

import cudnn
from frost_test_utils import _SM, arch_known_to_the_dsl, assert_no_new_spills, nvdisasm_candidates, requires_dsl, requires_rubin, select_engine

pytestmark = [pytest.mark.L0, requires_dsl]

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
_KERNEL_FILES = {"f16": "sm107/bprop_d256_f16.py", "fp8": "sm107/bprop_d256_fp8.py"}
_TEMPLATE_TAGS = {"f16": "sdpa_bwd_sm107_main_f16", "fp8": "sdpa_bwd_sm107_main_fp8"}
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
    for a literal keyword value (``k_dim=1``) must not see the docstrings and error messages that discuss it."""
    import io
    import tokenize

    lines = src.splitlines(keepends=True)
    offs, acc = [], 0
    for ln in lines:
        offs.append(acc)
        acc += len(ln)
    out = list(src)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            a = offs[tok.start[0] - 1] + tok.start[1]
            b = offs[tok.end[0] - 1] + tok.end[1]
            for i in range(a, b):
                if out[i] != "\n":
                    out[i] = " "
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
    assert not c.thd and not c.thd_declared_totals and not c.cu_seq_len
    assert not c.bias and not c.dbias
    assert not c.decode, "prefill bodies: a 128-row q tile per iteration; s_q == 1 is out of scope"
    assert c.layouts == frozenset({"bshd"})
    assert not c.tile_ms and not c.tile_ns, "fixed geometry: no tile axis, the heuristics list {} as the complete record"
    for deferred in ("padded", "sink", "dsink", "deterministic"):
        assert not getattr(c, deferred), f"{deferred} is deferred to PR-2c (plan Q4): claim it together with its accept test here and the tracker line"
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
        t["seq_len_q"] = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        t["seq_len_kv"] = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
        kw.update(use_padding_mask=True, seq_len_q=t["seq_len_q"], seq_len_kv=t["seq_len_kv"])
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


def test_reject_thd(monkeypatch):
    """Packed / ragged Q/K/V: no THD lowering on this row (plan PR-5)."""
    assert _decline_reason(monkeypatch, thd=True) is not None


def test_reject_decode_shaped(monkeypatch):
    assert _decline_reason(monkeypatch, sq=1, skv=256) is not None


def test_reject_non_bshd_layout(monkeypatch):
    """A BHSD-contiguous Q is declined by the layout envelope (``layouts == {"bshd"}``): the body derives its
    head / batch strides from BSHD-compact storage.  (Staging of an odd dO is the adapter's business, not the row's.)"""
    assert _decline_reason(monkeypatch, stride_fn=_bhsd_stride) is not None


def test_padding_mask_follows_the_padded_claim(monkeypatch):
    """Per-batch KV lengths.  Deferred in v1 (the row pins ``padded=False`` above) -> a REAL graph with a padding mask
    is declined; once the claim flips, this test inverts (the graph is served) and the poisoned dead-entry case below
    takes over the numerics -- inverted rather than deleted so the claim keeps a test on this side of the row."""
    reason = _decline_reason(monkeypatch, padded=True)
    if _spec().capabilities.padded:
        assert reason is None, reason
    else:
        assert reason is not None


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


def _causal_keep(sq, skv, dev="cuda", bottom_right=False, left=None, right=0):
    qi = torch.arange(sq, device=dev).view(-1, 1)
    ki = torch.arange(skv, device=dev).view(1, -1)
    diag = (skv - sq) if bottom_right else 0
    keep = ki <= qi + diag + right
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


def _reference64(q, k, v, do, keep=None, group=1):
    """fp64 attention backward on the STORAGE-rounded operands.  ``keep`` is a bool mask broadcastable to
    [B, 1, S_q, S_kv].  GQA groups q heads CONTIGUOUSLY (kv head h serves q heads h*g .. h*g+g-1), the convention the
    analyzer, the adapters and ``sdpa.fp8_ref.gqa_kv_head`` share."""
    q64, k64, v64, do64 = (x.double() for x in (q, k, v, do))
    kx = k64.repeat_interleave(group, dim=1) if group > 1 else k64
    vx = v64.repeat_interleave(group, dim=1) if group > 1 else v64
    scale = 1.0 / math.sqrt(q.shape[3])
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


def _run(b=2, hq=2, hkv=None, sq=512, skv=512, dt=torch.bfloat16, keep=None, omit_scale=False, seed=0, poison=None, runs=1, seq_lens=None, **sdpa_kwargs):
    """Build, PIN the engine, execute ``runs`` times and hand back every run's (dQ, dK, dV) plus the fp64 oracle.
    Inputs are unit normal (the pre-port driver's data), drawn on a CPU generator so the dataset does not depend on the
    GPU's SM count (torch's CUDA Philox lays draws out by grid size).  ``poison`` pre-fills the outputs before every run
    (a zero-initialized output hides a skipped store).  ``seq_lens=(seq_q_lens, seq_kv_lens)`` adds the padding mask."""
    hkv = hq if hkv is None else hkv
    group = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s, h):
        return torch.randn(bb, s, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do = draw(b, sq, hq), draw(b, sq, hq)
    k, v = draw(b, skv, hkv), draw(b, skv, hkv)
    if seq_lens is not None:
        keep = _padded_keep(sq, skv, *seq_lens) if keep is None else (keep & _padded_keep(sq, skv, *seq_lens))
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    if seq_lens is not None:
        sdpa_kwargs["padded"] = True
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dt=dt, scale=None if omit_scale else "default", **sdpa_kwargs)
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
    """attn_scale is OPTIONAL on the graph; omitting it must mean 1/sqrt(d), which the oracle assumes either way."""
    _run(omit_scale=True).check()


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


@requires_rubin
def test_dead_padded_entry_is_exactly_zero_when_padded_is_claimed():
    """The padded arm's degenerate input: one batch entry with seq_kv_len == 0.  Runs only once the row claims
    ``padded`` (until then ``test_padding_mask_follows_the_padded_claim`` asserts the decline); the pre-port fp8 d256
    FORWARD hangs on exactly this case, so the claim is gated on it.  Poisoned outputs: the dead entry's dK / dV / dQ
    must be exactly 0 (a select, not residue * 0), the live entry exact."""
    if not _spec().capabilities.padded:
        pytest.skip("padded is deferred (plan Q4); the decline is asserted host-side")
    b, sq, skv = 2, 256, 512
    run = _run(b=b, hq=2, sq=sq, skv=skv, seq_lens=([sq, sq], [skv, 0]), poison=float("nan")).check()
    for name, got in zip(("dQ", "dK", "dV"), run.outs[0]):
        dead = got[1].float()
        assert torch.isfinite(dead).all() and (dead == 0).all(), f"{name}: the seq_kv_len == 0 entry must be EXACTLY zero"


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
    Unlike the d512 band, d256 bf16 has a native competitor (backend engine 17, which forces its deterministic flag),
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


def _adapter(cls, b=2, hq=2, hkv=None, sq=512, skv=512, dt=torch.bfloat16, grad_dt=None, **kw):
    from cudnn.api_base import TensorDesc

    hkv = hq if hkv is None else hkv
    grad_dt = dt if grad_dt is None else grad_dt
    stats = TensorDesc(
        dtype=torch.float32, shape=(b, hq, sq, 1), stride=(hq * sq, sq, 1, 1), stride_order=(3, 2, 1, 0), device=torch.device("cuda", 0), name="stats"
    )
    return cls(
        sample_q=_adapter_desc((b, hq, sq, _D), dt, "q"),
        sample_k=_adapter_desc((b, hkv, skv, _D), dt, "k"),
        sample_v=_adapter_desc((b, hkv, skv, _D), dt, "v"),
        sample_o=_adapter_desc((b, hq, sq, _D), dt, "o"),
        sample_do=_adapter_desc((b, hq, sq, _D), dt, "dO"),
        sample_stats=stats,
        sample_dq=_adapter_desc((b, hq, sq, _D), grad_dt, "dQ"),
        sample_dk=_adapter_desc((b, hkv, skv, _D), grad_dt, "dK"),
        sample_dv=_adapter_desc((b, hkv, skv, _D), grad_dt, "dV"),
        scale_softmax=1.0 / math.sqrt(_D),
        **kw,
    )


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
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl_sm107 import _stage3_params
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE

    dk, dq = _stage3_params(DTYPE_BF16, causal=True, shift=512, gran=256, trim=True)
    assert (dk.a_is_m_major, dk.causal_mode) == (False, CAUSAL_K_LO)
    assert (dq.a_is_m_major, dq.causal_mode) == (True, CAUSAL_K_HI)
    assert dk.causal_shift == dq.causal_shift == 512 and dk.causal_gran == dq.causal_gran == 256
    assert dk.b_is_n_major and dq.b_is_n_major and dk.dtype_qkv == DTYPE_BF16
    for causal, trim in ((True, False), (False, True), (False, False)):
        dk, dq = _stage3_params(DTYPE_BF16, causal=causal, shift=0, gran=256, trim=trim)
        assert dk.causal_mode == dq.causal_mode == CAUSAL_K_NONE, (causal, trim)
        assert (dk.a_is_m_major, dq.a_is_m_major) == (False, True), "the majors are a layout fact, not a mask fact"


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
    assert api._b_chunk * api._qh_chunk * api._skv_pad * api._sq_pad * 2 <= 4 << 30 or api._qh_chunk == api._gqa_group


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
        (dict(seq_kv_lens_present=True), "padding masks"),
        (dict(thd=True), "THD"),
    ],
    ids=["fp32", "gqa-ratio", "decode", "right-band", "swa-zero", "br-without-causal", "deterministic", "padded", "thd"],
)
def test_half_adapter_backstop_refuses_what_the_row_declines(kw, needle):
    """Reaching one of these raises means the row lied; each is a ValueError naming the reason, never an assert."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    with pytest.raises(ValueError, match=needle):
        _adapter(SdpaBwdDslSm107, **kw).check_support()


def test_fp8_adapter_backstop_and_workspace():
    """The fp8 row's twin: E4M3 payloads with e4m3 / bf16 / fp16 gradients pass; E5M2, a mixed gradient triple, a half
    O payload and an off-contract gradient dtype raise.  Its workspace adds the bf16 Q / K copies, the three bf16
    partials and the amax scratch, and never chunks the batch."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    e4m3 = torch.float8_e4m3fn
    for grad in (e4m3, torch.bfloat16, torch.float16):
        api = _adapter(SdpaBwdDslSm107Fp8, b=2, hq=8, hkv=2, sq=257, skv=129, dt=e4m3, grad_dt=grad, is_causal=True)
        assert api.check_support()
        names = [n for n, _n, _d in api._scratch_plan()]
        for n in ("dv_part", "dk_part", "dq_ws", "q_bf16", "k_bf16", "amax_scratch", "q_pad", "k_pad", "lse_pad"):
            assert n in names, n
        assert api._b_chunk == 2, "the fp8 body has no batch_base: the whole batch is in-grid"
        assert api.scratch_workspace_bytes() == api.scratch_workspace_bytes() > 0
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


@requires_rubin
# S_q > S_kv is TOP-LEFT only: the frontend validator rejects a bottom-right causal graph with max_s_q > max_s_kv
# ("virtually slice the Q tensor") before any engine runs, so that combination is not a shape this row can be asked for.
@pytest.mark.parametrize("sq,skv,bottom_right", [(512, 512, False), (512, 1024, True), (768, 512, False)], ids=["square", "br-rect-kv", "tl-rect-q"])
def test_stage3_causal_k_trim_is_bitwise_the_untrimmed_rendering(monkeypatch, sq, skv, bottom_right):
    """The dK / dQ GEMMs' causal K-trim (plan s5: LO on dK, HI on dQ, over the kv-major workspace) is an OPTIMIZATION,
    made exact by the zero-filled workspace: rendering both GEMMs untrimmed (``STAGE3_CAUSAL_TRIM = False``, every
    k tile read) must give the SAME BITS for dQ / dK / dV.  A trim mode paired with the wrong operand major, or a
    shift off by the bottom-right diagonal, drops real tiles here and shows up as a non-zero diff."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(use_causal_mask_bottom_right=True) if bottom_right else dict(use_causal_mask=True)
    keep = _causal_keep(sq, skv, bottom_right=bottom_right)
    trimmed = _run(b=2, hq=4, hkv=2, sq=sq, skv=skv, keep=keep, poison=float("nan"), **kw).check()
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    untrimmed = _run(b=2, hq=4, hkv=2, sq=sq, skv=skv, keep=keep, poison=float("nan"), **kw).check()
    for name, x, y in zip(("dQ", "dK", "dV"), trimmed.outs[0], untrimmed.outs[0]):
        n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
        assert n_diff == 0, f"{name}: trimmed vs untrimmed stage 3 differ in {n_diff} elements (max|diff|={(x.float() - y.float()).abs().max().item():.3e})"


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
_RING_WAIT_SITES = {"f16": (23, 15), "fp8": (17, 24)}  # family -> (ring sites, idle sites); fp8 measured on 5e99bb9b, f16 after the P8 / drain fixes
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
    """``FenceCode`` pragmas and ``-uumn`` are INTERNAL-ONLY debugging tools: never in a production kernel, a commit or a
    PR (user rule 2026-09-22); a ``cfence`` is a scheduling pin, not a memory fence."""
    src = _kernel_source(family)
    assert not re.search(r"uumn|FenceCode|cfence", src), f"{family}: an internal ptxas scheduling knob is in the kernel"


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
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams
    FILES = {"f16": "sm107/bprop_d256_f16.py", "fp8": "sm107/bprop_d256_fp8.py"}
    MASKS = {"dense": {}, "causal": dict(window_right=0), "causal_swa": dict(window_right=0, window_left=640)}
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
    print("SASS USETMAXREG", cnt("USETMAXREG"))
    print("SASS STL", cnt("STL"))
    print("SASS LDL", cnt("LDL"))
    print("SASS MEMBAR_GPU", cnt("MEMBAR.ALL.GPU"))
    print("SASS CGAERRBAR", cnt("CGAERRBAR"))
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
    pytest.param("fp8", "dense", id="fp8-dense"),
    pytest.param("fp8", "causal", id="fp8-causal"),
]
# The masked rows of the above: the mask form pin (rules/frost-tile-dsl.md s10d) applies to them only.
_MASKED_SASS_PIN_ROWS = [r for r in _SASS_PIN_ROWS if r.values[1] != "dense"]
# Spill bounds: the counts MEASURED on the branch's own toolchain (cutlass-dsl 4.8.0 + the internal CUDA toolkit's ptxas,
# 2026-09-23, B=1 H=8 S=1024): 0 / 0 STL / LDL on all three rows -- the plan's target (s10.9) -- plus the DSL / ptxas
# jitter frost_test_utils.SPILL_TOLERANCE allows.  Never loosen a row to turn it green; a real spill adds tens.
_SPILL_PINS = {
    ("f16", "dense"): {"STL": 0, "LDL": 0},
    ("f16", "causal"): {"STL": 0, "LDL": 0},
    ("fp8", "dense"): {"STL": 0, "LDL": 0},
    ("fp8", "causal"): {"STL": 0, "LDL": 0},
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
    after).  ``R2P == 0`` on a masked build is the detector the rule names; both bodies spell the form as ``MASK_FORM``."""
    stats, _order = _sass_probe(tmp_path, family, mask)
    assert stats["R2P"] > 0, f"{family} {mask}: no R2P in the masked build -- the mask arm is the per-cell compare + select form"
    mod = _load_kernel(family)
    assert mod.MASK_FORM == "bits", f"{mod.__name__}: MASK_FORM must be the bit-word form"
