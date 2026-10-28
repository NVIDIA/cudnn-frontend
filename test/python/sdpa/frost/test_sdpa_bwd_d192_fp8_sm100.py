# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""``sdpa_bwd_sm100_d192_fp8``: dense D_QK = 192 / D_V = 128 per-tensor FP8 backward on SM100 / SM103."""

import math

import cudnn
import pytest
import torch

import test_sdpa_bwd_fp8_sm107 as _rubin
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm100_d192_fp8"
_D, _DV = 192, 128
_SM100_CC = (10, 0)
_SM103_CC = (10, 3)
_GRAD_DTYPES = [torch.float8_e4m3fn, torch.bfloat16]


@pytest.fixture
def sm100_row(monkeypatch):
    """Point the Rubin fp8 suite's helpers at this row."""
    monkeypatch.setattr(_rubin, "_ENGINE", _ENGINE)
    monkeypatch.setattr(_rubin, "_D", _D)
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
    g = _rubin._build_graph(**{"d": _D, "d_v": _DV, **kw})
    facts = g._facts_for(engines.ga.analyze)
    facts = facts.__class__(**{**facts.__dict__, "device_cc": cc}) if hasattr(facts, "__dict__") else facts
    return engines.mismatch(_spec().capabilities, facts)


def test_engine_is_registered():
    c = _spec().capabilities
    assert c.is_fp8 and not c.is_mxfp8
    assert c.d == frozenset({128, 192}) and c.dqk_ge_dv and not c.d_envelope
    assert c.dtypes == frozenset({cudnn.data_type.FP8_E4M3})
    assert c.out_dtypes == frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    assert c.amax_dgrad and c.causal and not c.gqa and not c.swa and not c.deterministic and not c.bottom_right and not c.thd
    assert (c.sm_lo, c.sm_hi) == (100, 106)


def test_row_ranks_right_behind_the_half_row():
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    names = [s.name for s in ENGINE_SPECS]
    assert names.index("sdpa_bwd_sm100_d192") + 1 == names.index(_ENGINE) < names.index("sdpa_bwd_sm100_d256")


@pytest.mark.parametrize("cc", [_SM100_CC, _SM103_CC], ids=["sm100", "sm103"])
def test_row_admits_dense_d192_e4m3(monkeypatch, cc):
    assert _mismatch(monkeypatch, cc=cc) is None
    assert _mismatch(monkeypatch, cc=cc, causal=False) is None
    assert _mismatch(monkeypatch, cc=cc, grad_dtypes=(cudnn.data_type.BFLOAT16,) * 3) is None
    assert _mismatch(monkeypatch, cc=cc, request_amax=()) is None


@pytest.mark.parametrize(
    "kw,needle",
    [
        (dict(d=128, d_v=128), ""),
        (dict(d=256, d_v=256), ""),
        (dict(cc=(10, 7)), "requires SM100-106"),
        (dict(cc=(12, 0)), "requires SM100-106"),
        (dict(h=4, hkv=2), ""),
        (dict(left_bound=64), ""),
        (dict(deterministic=True), ""),
        (dict(bottom_right=True), ""),
    ],
    ids=["d128", "d256", "rubin", "sm120", "gqa", "window", "deterministic", "bottom-right"],
)
def test_row_declines(monkeypatch, kw, needle):
    reason = _mismatch(monkeypatch, **kw)
    assert reason is not None and needle.lower() in reason.lower(), reason


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
def test_dense(sm100_row, grad_dtype):
    sm100_row._run_fp8(b=1, hq=2, sq=512, skv=512, causal=False, grad_dtype=grad_dtype, d_v=_DV).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
def test_causal(sm100_row, grad_dtype):
    sm100_row._run_fp8(b=2, hq=3, sq=768, skv=768, causal=True, grad_dtype=grad_dtype, d_v=_DV).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("sq,skv", [(768, 1280), (500, 500), (257, 129), (129, 1), (1, 300)], ids=["rect", "500", "257x129", "129x1", "1x300"])
def test_non_tile_multiple_seqlens(sm100_row, sq, skv):
    sm100_row._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=False, d_v=_DV).check()
    sm100_row._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=True, d_v=_DV).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("request_amax", [(), ("amax_dP",), ("amax_dQ", "amax_dV")], ids=["none", "dP", "dQ-dV"])
def test_amax_subsets(sm100_row, request_amax):
    sm100_row._run_fp8(b=1, hq=2, sq=512, skv=640, causal=True, request_amax=request_amax, d_v=_DV).check()


@requires_pre_rubin_blackwell
def test_repeated_launches_are_bitwise(sm100_row):
    run = sm100_row._run_fp8(b=1, hq=4, sq=1024, skv=1024, causal=True, runs=3, poison=float("nan"), d_v=_DV).check()
    for later in run.outs[1:]:
        for name in ("dQ", "dK", "dV"):
            assert torch.equal(run.outs[0][name].view(torch.int8), later[name].view(torch.int8)), name
    for later in run.amax[1:]:
        for name, value in run.amax[0].items():
            assert torch.equal(value, later[name]), name


def test_pair_host_binds_one_kernel_pair_per_specialization(monkeypatch):
    import cutlass

    from cudnn.sdpa.bwd.kernels.sm100 import prepared_d192_fp8_host as pair

    compiled = []

    def record(host, *args, **kwargs):
        compiled.append(host)
        return object()

    mx_w0, mx_w128 = (128, 128, False, True, 0), (128, 128, False, True, 128)
    fp8_sm103, fp8_sm100, mx_192 = (192, 128, False, False, 0), (192, 128, True, False, 0), (192, 128, False, True, 0)
    modules = {spec: pair._specialized(spec) for spec in (mx_w0, mx_w128, fp8_sm103, fp8_sm100, mx_192)}
    for module in modules.values():
        monkeypatch.setattr(module, "compile_cached", record)
    out = cutlass.BFloat16
    pair.compile_host_mx((), (), (), out, True, 103, 1, 128, 128, 0)
    pair.compile_host_mx((), (), (), out, True, 103, 1, 128, 128, 128)
    pair.compile_host_mx((), (), (), out, True, 103, 1, 128, 128, 0)
    pair.compile_host_mx((), (), (), out, True, 103, 1, 192, 128, 0)
    pair.compile_host((), (), out, True, 103, (False,) * 4, 192, 128, 1)
    pair.compile_host((), (), out, True, 100, (False,) * 4, 192, 128, 1)
    assert len(compiled) == 6
    for spec, module in modules.items():
        assert module.SPEC == spec and (module.dkdv_host, module.dq_host) == pair._kernel_hosts(spec)
    assert modules[mx_w0].dq_host is not modules[mx_w128].dq_host
    assert modules[fp8_sm103].dkdv_host is not modules[mx_192].dkdv_host
    assert modules[fp8_sm103].dkdv_host is not modules[fp8_sm100].dkdv_host


@requires_pre_rubin_blackwell
def test_causal_empty_kv_tiles(sm100_row):
    sm100_row._run_fp8(b=4, hq=32, sq=256, skv=1024, causal=True, d_v=_DV).check()


@requires_pre_rubin_blackwell
def test_negative_scale_declined_at_check():
    from cudnn.sdpa.bwd.api_dsl_sm100_d192_fp8 import SdpaBwdDslSm100D192Fp8

    b, h, s = 1, 2, 256

    def bshd(width, dtype):
        return torch.empty(b, s, h, width, dtype=dtype, device="cuda").permute(0, 2, 1, 3)

    stats = torch.empty(b, h, s, 1, dtype=torch.float32, device="cuda")
    samples = {n: bshd(_D if n in ("q", "k") else _DV, torch.float8_e4m3fn) for n in ("q", "k", "v", "o", "do")}
    samples.update(stats=stats, **{n: bshd(_D if n in ("dq", "dk") else _DV, torch.bfloat16) for n in ("dq", "dk", "dv")})

    def api(scale):
        return SdpaBwdDslSm100D192Fp8(**{"sample_" + n: t for n, t in samples.items()}, is_causal=True, scale_softmax=scale)

    assert api(0.1).check_support()
    with pytest.raises(NotImplementedError, match="finite and positive"):
        api(-0.1).check_support()


@requires_pre_rubin_blackwell
def test_explicit_attn_scale(sm100_row):
    sm100_row._run_fp8(b=1, hq=2, sq=384, skv=512, causal=True, attn_scale=0.5 / math.sqrt(_D), d_v=_DV).check()
