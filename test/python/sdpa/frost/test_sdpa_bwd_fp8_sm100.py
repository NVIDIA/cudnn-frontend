# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""``sdpa_bwd_sm100_d128_fp8``: dense d128 per-tensor FP8 backward on SM100 / SM103."""

import math

import cudnn
import pytest
import torch

import test_sdpa_bwd_fp8_sm107 as _rubin
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm100_d128_fp8"
_D = 128
_SM100_CC = (10, 0)
_SM103_CC = (10, 3)
_GRAD_DTYPES = [torch.float8_e4m3fn, torch.bfloat16]


@pytest.fixture
def sm100_row(monkeypatch):
    """Point the Rubin fp8 suite's helpers at this row."""
    monkeypatch.setattr(_rubin, "_ENGINE", _ENGINE)
    monkeypatch.setattr(_rubin, "_D", _D)
    return _rubin


@pytest.fixture
def sm100_row_d64(monkeypatch):
    """The same row at head dimension 64."""
    monkeypatch.setattr(_rubin, "_ENGINE", _ENGINE)
    monkeypatch.setattr(_rubin, "_D", 64)
    return _rubin


def _spec():
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == _ENGINE), None)
    assert spec is not None
    return spec


def _mismatch(monkeypatch, cc=_SM100_CC, **kw):
    from cudnn.sdpa.bwd import engines

    monkeypatch.setattr(_rubin, "_D", _D)
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.1")))
    monkeypatch.setattr(engines, "cutedsl_too_old", lambda v: False)
    monkeypatch.setattr(engines, "cutedsl_arch_requirement_error", lambda cc: None)
    g = _rubin._build_graph(**{"d": _D, **kw})
    facts = g._facts_for(engines.ga.analyze)
    facts = facts.__class__(**{**facts.__dict__, "device_cc": cc}) if hasattr(facts, "__dict__") else facts
    return engines.mismatch(_spec().capabilities, facts)


def test_engine_is_registered():
    c = _spec().capabilities
    assert c.is_fp8 and not c.is_mxfp8
    assert c.d == frozenset({64, 128}) and not c.d_envelope and not c.dqk_ge_dv
    assert c.dtypes == frozenset({cudnn.data_type.FP8_E4M3})
    assert c.out_dtypes == frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    assert c.amax_dgrad and c.causal and c.gqa and c.swa and not c.deterministic and not c.bottom_right and not c.thd
    assert (c.sm_lo, c.sm_hi) == (100, 106)


@pytest.mark.parametrize("cc", [_SM100_CC, _SM103_CC], ids=["sm100", "sm103"])
def test_row_admits_dense_d128_e4m3(monkeypatch, cc):
    assert _mismatch(monkeypatch, cc=cc) is None
    assert _mismatch(monkeypatch, cc=cc, d=64) is None
    assert _mismatch(monkeypatch, cc=cc, d=64, left_bound=127) is None
    assert _mismatch(monkeypatch, cc=cc, causal=False) is None
    assert _mismatch(monkeypatch, cc=cc, grad_dtypes=(cudnn.data_type.BFLOAT16,) * 3) is None
    assert _mismatch(monkeypatch, cc=cc, request_amax=()) is None


@pytest.mark.parametrize(
    "kw,needle",
    [
        (dict(d=256), "serves D in [64, 128]"),
        (dict(cc=(10, 7)), "requires SM100-106"),
        (dict(cc=(12, 0)), "requires SM100-106"),
        (dict(hkv=3), "GQA"),
        (dict(left_bound=64), "sliding window at d64 only"),
        (dict(deterministic=True), "deterministic"),
        (dict(bottom_right=True), "bottom-right"),
    ],
    ids=["d256", "rubin", "sm120", "gqa", "window", "deterministic", "bottom-right"],
)
def test_row_declines(monkeypatch, kw, needle):
    reason = _mismatch(monkeypatch, **kw)
    assert reason is not None and needle.lower() in reason.lower(), reason


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
def test_dense(sm100_row, grad_dtype):
    sm100_row._run_fp8(b=1, hq=2, sq=512, skv=512, causal=False, grad_dtype=grad_dtype).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
def test_causal(sm100_row, grad_dtype):
    sm100_row._run_fp8(b=2, hq=3, sq=768, skv=768, causal=True, grad_dtype=grad_dtype).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("sq,skv", [(768, 1280), (500, 500), (257, 129), (129, 1), (1, 300)], ids=["rect", "500", "257x129", "129x1", "1x300"])
def test_non_tile_multiple_seqlens(sm100_row, sq, skv):
    sm100_row._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=False).check()
    sm100_row._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=True).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("request_amax", [(), ("amax_dP",), ("amax_dQ", "amax_dV")], ids=["none", "dP", "dQ-dV"])
def test_amax_subsets(sm100_row, request_amax):
    sm100_row._run_fp8(b=1, hq=2, sq=512, skv=640, causal=True, request_amax=request_amax).check()


@requires_pre_rubin_blackwell
def test_repeated_launches_are_bitwise(sm100_row):
    run = sm100_row._run_fp8(b=1, hq=4, sq=1024, skv=1024, causal=True, runs=3, poison=float("nan")).check()
    for later in run.outs[1:]:
        for name in ("dQ", "dK", "dV"):
            assert torch.equal(run.outs[0][name].view(torch.int8), later[name].view(torch.int8)), name
    for later in run.amax[1:]:
        for name, value in run.amax[0].items():
            assert torch.equal(value, later[name]), name


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
@pytest.mark.parametrize("hq,hkv,causal", [(4, 2, False), (4, 2, True), (8, 1, True), (16, 2, False)], ids=["r2-dense", "r2-causal", "r8-mqa", "r8-dense"])
def test_gqa(sm100_row, hq, hkv, causal, grad_dtype):
    sm100_row._run_fp8(b=2, hq=hq, hkv=hkv, sq=640, skv=512, causal=causal, grad_dtype=grad_dtype).check()


@requires_pre_rubin_blackwell
def test_gqa_repeated_launches_are_bitwise(sm100_row):
    run = sm100_row._run_fp8(b=1, hq=8, hkv=2, sq=1024, skv=1024, causal=True, runs=2, poison=float("nan")).check()
    for name in ("dQ", "dK", "dV"):
        assert torch.equal(run.outs[0][name].view(torch.int8), run.outs[1][name].view(torch.int8)), name
    for name, value in run.amax[0].items():
        assert torch.equal(value, run.amax[1][name]), name


@requires_pre_rubin_blackwell
def test_explicit_attn_scale(sm100_row):
    sm100_row._run_fp8(b=1, hq=2, sq=384, skv=512, causal=True, attn_scale=0.5 / math.sqrt(_D)).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_d64(sm100_row_d64, grad_dtype, causal):
    sm100_row_d64._run_fp8(b=2, hq=3, sq=640, skv=512, causal=causal, grad_dtype=grad_dtype).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("sq,skv", [(500, 500), (257, 129), (129, 1)], ids=["500", "257x129", "129x1"])
def test_d64_non_tile_multiple_seqlens(sm100_row_d64, sq, skv):
    sm100_row_d64._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=False).check()
    sm100_row_d64._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=True).check()


@requires_pre_rubin_blackwell
def test_d64_gqa_amax_bitwise(sm100_row_d64):
    run = sm100_row_d64._run_fp8(b=1, hq=8, hkv=2, sq=1024, skv=1024, causal=True, runs=2, poison=float("nan")).check()
    for name in ("dQ", "dK", "dV"):
        assert torch.equal(run.outs[0][name].view(torch.int8), run.outs[1][name].view(torch.int8)), name
    for name, value in run.amax[0].items():
        assert torch.equal(value, run.amax[1][name]), name


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
@pytest.mark.parametrize(
    "causal,left", [(True, 127), (False, 127), (True, 300), (True, 32)], ids=["causal-w128", "window-only-w128", "causal-w301", "causal-w33"]
)
def test_d64_sliding_window(sm100_row_d64, grad_dtype, causal, left):
    sm100_row_d64._run_fp8(b=2, hq=2, sq=1024, skv=1024, causal=causal, left=left, grad_dtype=grad_dtype).check()


@requires_pre_rubin_blackwell
def test_d64_sliding_window_gqa_ragged(sm100_row_d64):
    sm100_row_d64._run_fp8(b=1, hq=4, hkv=2, sq=700, skv=900, causal=True, left=127).check()
