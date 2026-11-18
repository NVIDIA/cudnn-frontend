# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""``sdpa_bwd_sm100_d192_fp8`` dK/dV + dQ kernel pair checks; ``sm100_row`` points the fp8 suite's helpers at the row."""

import math

import pytest
import torch

_D, _DV = 192, 128
_GRAD_DTYPES = [torch.float8_e4m3fn, torch.bfloat16]


@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
def test_dense(sm100_row, grad_dtype):
    sm100_row._run_fp8(b=1, hq=2, sq=512, skv=512, causal=False, grad_dtype=grad_dtype, d_v=_DV).check()


@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES, ids=["e4m3", "bf16"])
def test_causal(sm100_row, grad_dtype):
    sm100_row._run_fp8(b=2, hq=3, sq=768, skv=768, causal=True, grad_dtype=grad_dtype, d_v=_DV).check()


@pytest.mark.parametrize("sq,skv", [(768, 1280), (500, 500), (257, 129), (129, 1), (1, 300)], ids=["rect", "500", "257x129", "129x1", "1x300"])
def test_non_tile_multiple_seqlens(sm100_row, sq, skv):
    sm100_row._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=False, d_v=_DV).check()
    sm100_row._run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=True, d_v=_DV).check()


@pytest.mark.parametrize("request_amax", [(), ("amax_dP",), ("amax_dQ", "amax_dV")], ids=["none", "dP", "dQ-dV"])
def test_amax_subsets(sm100_row, request_amax):
    sm100_row._run_fp8(b=1, hq=2, sq=512, skv=640, causal=True, request_amax=request_amax, d_v=_DV).check()


def test_repeated_launches_are_bitwise(sm100_row):
    run = sm100_row._run_fp8(b=1, hq=4, sq=1024, skv=1024, causal=True, runs=3, poison=float("nan"), d_v=_DV).check()
    for later in run.outs[1:]:
        for name in ("dQ", "dK", "dV"):
            assert torch.equal(run.outs[0][name].view(torch.int8), later[name].view(torch.int8)), name
    for later in run.amax[1:]:
        for name, value in run.amax[0].items():
            assert torch.equal(value, later[name]), name


def test_causal_empty_kv_tiles(sm100_row):
    sm100_row._run_fp8(b=4, hq=32, sq=256, skv=1024, causal=True, d_v=_DV, runs=32).check()


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


def test_explicit_attn_scale(sm100_row):
    sm100_row._run_fp8(b=1, hq=2, sq=384, skv=512, causal=True, attn_scale=0.5 / math.sqrt(_D), d_v=_DV).check()
